"""ZIP and ZIPS OpenEXR read and Deflate control lane."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np

from pixtreme._io.formats.exr.container import (
    _ExrChannel,
    _ExrChunk,
    _ExrContainer,
    _ExrReadChunks,
    _gpu_error,
)
from pixtreme._io.formats.exr.packing import (
    _numpy_offsets,
    _nvcomp_deflate_codec,
    _read_worker_count,
    _restore_exr_gpu_chunks,
    _unpack_exr_output,
)

_ZIPS_READ_WORKER_LIMIT = 4


def _strict_zlib_payload(
    chunk: _ExrChunk,
    deflate: bytes,
    expected_adler: int,
) -> tuple[bytes, int, bytes]:
    decompressor = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
    try:
        host_decoded = decompressor.decompress(deflate, chunk.expected_size + 1)
    except zlib.error as error:
        raise _gpu_error(
            why="the EXR zlib wrapper contains an invalid raw Deflate payload",
            what=f"chunk_y={chunk.y}, error={error}",
            how="encode one complete RFC 1951 stream between the zlib header and Adler-32 trailer",
        ) from error
    if len(host_decoded) != chunk.expected_size or decompressor.unconsumed_tail:
        raise _gpu_error(
            why="the host-validated raw Deflate output sizes differ from the EXR container descriptor",
            what=f"chunk_y={chunk.y}, actual={len(host_decoded)}, expected={chunk.expected_size}",
            how="verify the Deflate payload and the channel-derived uncompressed chunk size",
        )
    if not decompressor.eof:
        raise _gpu_error(
            why="the EXR raw Deflate payload ends before its final block",
            what=f"chunk_y={chunk.y}, deflate_size={len(deflate)}",
            how="provide a complete RFC 1951 stream before the Adler-32 trailer",
        )
    if decompressor.unused_data:
        raise _gpu_error(
            why="the EXR raw Deflate payload has trailing bytes before the Adler-32 trailer",
            what=f"chunk_y={chunk.y}, trailing_size={len(decompressor.unused_data)}",
            how="place the Adler-32 trailer immediately after the final Deflate block",
        )
    observed_adler = zlib.adler32(host_decoded)
    if observed_adler != expected_adler:
        raise _gpu_error(
            why="the host Adler-32 result does not match the EXR zlib trailer",
            what=f"chunk_y={chunk.y}, observed=0x{observed_adler:08x}, expected=0x{expected_adler:08x}",
            how="verify that the zlib payload and big-endian Adler-32 trailer are complete and unmodified",
        )
    return deflate, expected_adler, host_decoded


def _zlib_payload(container: _ExrContainer, chunk: _ExrChunk) -> tuple[bytes, int, bytes]:
    payload = container.data[chunk.payload_start : chunk.payload_end]
    if len(payload) < 6:
        raise _gpu_error(
            why="the EXR zlib wrapper is truncated before its Deflate payload and Adler-32 trailer",
            what=f"chunk_y={chunk.y}, packed_size={len(payload)}",
            how="provide a complete RFC 1950 zlib stream or a raw-stored chunk",
        )
    cmf, flg = payload[0], payload[1]
    if (cmf & 0x0F) != 8 or (cmf >> 4) > 7 or ((cmf << 8) | flg) % 31:
        raise _gpu_error(
            why="the EXR zlib header has an invalid compression method, window, or FCHECK",
            what=f"chunk_y={chunk.y}, CMF=0x{cmf:02x}, FLG=0x{flg:02x}",
            how="encode a valid RFC 1950 Deflate stream with a correct header check",
        )
    if flg & 0x20:
        raise _gpu_error(
            why="the EXR zlib stream requests a preset dictionary",
            what=f"chunk_y={chunk.y}, FLG=0x{flg:02x}",
            how="encode EXR ZIP/ZIPS without an RFC 1950 preset dictionary",
        )
    deflate = payload[2:-4]
    if not deflate:
        raise _gpu_error(
            why="the EXR zlib wrapper contains no raw Deflate payload",
            what=f"chunk_y={chunk.y}, packed_size={len(payload)}",
            how="provide a complete RFC 1951 payload between the zlib header and trailer",
        )
    expected_adler = struct.unpack(">I", payload[-4:])[0]
    decompressor = zlib.decompressobj(wbits=zlib.MAX_WBITS)
    try:
        host_decoded = decompressor.decompress(payload, chunk.expected_size + 1)
    except zlib.error:
        return _strict_zlib_payload(chunk, deflate, expected_adler)
    if (
        len(host_decoded) != chunk.expected_size
        or decompressor.unconsumed_tail
        or not decompressor.eof
        or decompressor.unused_data
    ):
        return _strict_zlib_payload(chunk, deflate, expected_adler)
    return deflate, expected_adler, host_decoded


def _prepare_exr_read_chunks(
    container: _ExrContainer,
    *,
    include_staging: bool = True,
    pinned_decoded: bool = False,
) -> _ExrReadChunks:
    chunks = container.chunks
    decoded_sizes = np.fromiter((chunk.expected_size for chunk in chunks), dtype=np.int64, count=len(chunks))
    decoded_offsets = _numpy_offsets(decoded_sizes)
    compressed = np.fromiter((not chunk.raw_stored for chunk in chunks), dtype=np.uint8, count=len(chunks))
    compressed_indices = np.flatnonzero(compressed)
    decoded_payloads: dict[int, tuple[bytes, int, bytes]] = {}
    if compressed_indices.size:
        compressed_chunks = tuple(chunks[int(index)] for index in compressed_indices)
        worker_count = _read_worker_count(len(compressed_chunks))
        if container.compression == "zip":
            worker_count = min(worker_count, 8)
        if container.compression == "zips":
            worker_count = min(worker_count, _ZIPS_READ_WORKER_LIMIT)

            batch_size = (len(compressed_chunks) + worker_count - 1) // worker_count
            chunk_batches = tuple(
                compressed_chunks[start : start + batch_size] for start in range(0, len(compressed_chunks), batch_size)
            )

            def decode_batch(batch: Sequence[_ExrChunk]) -> tuple[tuple[bytes, int, bytes], ...]:
                return tuple(_zlib_payload(container, chunk) for chunk in batch)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                batch_results = executor.map(decode_batch, chunk_batches)
                results = tuple(result for batch in batch_results for result in batch)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = tuple(executor.map(lambda chunk: _zlib_payload(container, chunk), compressed_chunks))
        decoded_payloads = dict(zip((int(index) for index in compressed_indices), results, strict=True))

    stage_parts: list[bytes] = []
    stage_sizes = np.zeros(len(chunks), dtype=np.int64)
    expected_adler = np.zeros(len(chunks), dtype=np.uint32)
    decoded_byte_count = int(decoded_sizes.sum())
    host_decoded = (
        np.frombuffer(cp.cuda.alloc_pinned_memory(decoded_byte_count), dtype=np.uint8, count=decoded_byte_count)
        if pinned_decoded
        else np.empty(decoded_byte_count, dtype=np.uint8)
    )
    for index, chunk in enumerate(chunks):
        decoded_offset = int(decoded_offsets[index])
        decoded_size = int(decoded_sizes[index])
        if compressed[index]:
            stage_payload, adler, decoded_payload = decoded_payloads[index]
            expected_adler[index] = adler
        else:
            stage_payload = container.data[chunk.payload_start : chunk.payload_end]
            decoded_payload = stage_payload
        if len(decoded_payload) != decoded_size:
            raise _gpu_error(
                why="a prepared EXR chunk differs from its expected uncompressed size",
                what=f"chunk_y={chunk.y}, prepared={len(decoded_payload)}, expected={decoded_size}",
                how="store exactly the channel-derived raw scanline bytes",
            )
        if include_staging:
            stage_parts.append(stage_payload)
            stage_sizes[index] = len(stage_payload)
        host_decoded[decoded_offset : decoded_offset + decoded_size] = np.frombuffer(decoded_payload, dtype=np.uint8)
    stage_offsets = _numpy_offsets(stage_sizes)
    host_staging = np.frombuffer(b"".join(stage_parts), dtype=np.uint8)
    return _ExrReadChunks(
        host_staging=host_staging,
        host_decoded=host_decoded,
        stage_offsets=stage_offsets,
        stage_sizes=stage_sizes,
        decoded_offsets=decoded_offsets,
        decoded_sizes=decoded_sizes,
        compressed=compressed,
        expected_adler=expected_adler,
    )


def _decode_deflate_chunks(
    device_staging: cp.ndarray,
    input_ranges: Sequence[tuple[int, int]],
    decoded: cp.ndarray,
    output_ranges: Sequence[tuple[int, int]],
    *,
    verify_output_sizes: bool = True,
) -> None:
    from nvidia import nvcomp

    stream = cp.cuda.get_current_stream()
    inputs = nvcomp.as_arrays(
        [device_staging[offset : offset + size] for offset, size in input_ranges], cuda_stream=int(stream.ptr)
    )
    outputs = [decoded[offset : offset + size] for offset, size in output_ranges]
    codec = _nvcomp_deflate_codec(cp.cuda.Device().id, int(stream.ptr))
    if verify_output_sizes:
        actual_sizes = tuple(int(codec.get_uncomp_buffer_size(source)) for source in inputs)
        expected_sizes = tuple(size for _, size in output_ranges)
        if actual_sizes != expected_sizes:
            raise _gpu_error(
                why="nvCOMP raw Deflate output sizes differ from the EXR container descriptor",
                what=f"actual={actual_sizes!r}, expected={expected_sizes!r}",
                how="verify the Deflate payload and the channel-derived uncompressed chunk sizes",
            )
    codec.decode(inputs, out=outputs)


def _read_exr_zip_custom_cpu(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    *,
    output_dtype: str,
) -> cp.ndarray:
    prepared = _prepare_exr_read_chunks(container, include_staging=False, pinned_decoded=True)
    decoded = cp.asarray(prepared.host_decoded)
    cp.cuda.get_current_stream().synchronize()
    _restore_exr_gpu_chunks(
        decoded,
        prepared.decoded_offsets,
        prepared.decoded_sizes,
        prepared.compressed,
        None,
    )
    return _unpack_exr_output(
        container,
        selected,
        decoded,
        prepared.decoded_offsets,
        prepared.decoded_sizes,
        even_odd_grouped=prepared.compressed,
        output_dtype=output_dtype,
    )
