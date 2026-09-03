# Hardware video codec connections

pixtreme owns the boundary between GPU-resident image arrays and `Frame`. A hardware decoder or encoder owns its
session, compressed packets, codec configuration, and timestamps. Connect the two layers with an explicit raw-surface
format, colour signal, CUDA device, stream-ordering rule, and allocation lifetime. None of those properties should be
inferred from an array pointer alone.

## Choose the decoder entry point

Use the packed-format entry that describes the decoder surface. Both functions unpack to a C-contiguous, full-range
float32 YCbCr 4:4:4 `Frame` without a host raw-frame copy.

| Decoder surface | pixtreme entry | Required CuPy storage |
|---|---|---|
| NV12, 8 bit | `px.io.from_nv12` | C-contiguous 1D `uint8`: Y followed by interleaved Cb/Cr |
| P010, 10 bit | `px.io.from_p010` | C-contiguous 1D `uint16`: Y followed by interleaved Cb/Cr, ten bits MSB-aligned and the lower six bits padding |
| Already unpacked device array | `px.io.from_array` | HWC, NHWC, CHW, or NCHW CUDA DLPack producer or CuPy array |

`px.io.from_array` interprets an ordinary channel axis. It is not a substitute for unpacking a raw NV12 or P010
buffer. Some decoder APIs label a ten-bit MSB-aligned output allocation as `P016`; use `px.io.from_p010` only when the
decoder contract proves that the surface contains ten meaningful bits in P010 layout and that the lower six bits are
padding. A true 16-bit surface or an unknown packing does not satisfy the P010 contract.

The packed entries accept a CuPy array, not an arbitrary DLPack object. If the decoder exposes only DLPack, create the
CuPy view with `cp.from_dlpack` inside the intended consumer-stream context, keep the decoder's owner object alive, and
then pass the resulting C-contiguous 1D view to the packed entry.

## Map the decoded signal explicitly

Pass the signal observed from codec VUI, container `colr`, or another authoritative stream description on every
boundary where it is known:

| Stream signal | pixtreme claim | Effect |
|---|---|---|
| `matrix_coefficients` | `matrix=` | Stamps the YCbCr coefficient provenance; it does not change array values |
| `chroma_sample_loc_type` | `siting=` | Selects the 4:2:0 chroma phase used by upsampling or downsampling |
| `video_full_range_flag` | `range=` | Selects legal-code expansion or full-container scaling |
| colour primaries | `colorspace=` | States the RGB primary system associated with the signal |
| transfer characteristics | `gamma=` | States the transfer characteristic associated with the signal |

Common progressive 4:2:0 mappings are:

| H.273 value | pixtreme token |
|---|---|
| `matrix_coefficients = 1` | `matrix="BT.709"` |
| `matrix_coefficients = 5` or `6` | `matrix="BT.601"`, after the application confirms the equivalent non-constant-luminance coefficients |
| `matrix_coefficients = 9` | `matrix="BT.2020"` |
| `chroma_sample_loc_type = 0` | `siting="left"` |
| `chroma_sample_loc_type = 1` | `siting="center"` |
| `chroma_sample_loc_type = 2` | `siting="topleft"` |
| `video_full_range_flag = 0` | `range="legal"` |
| `video_full_range_flag = 1` | `range="full"` |

Do not infer chroma siting from matrix coefficients, colour primaries, resolution, or bit depth. `left` is pixtreme's
omission default, not evidence about a particular stream. Likewise, do not guess an unspecified matrix from chroma
siting. If signalling is absent or cannot be represented by pixtreme's closed tokens, reject the stream or apply a
documented application policy explicitly. The named-format functions support progressive H.273 siting types 0, 1,
and 2; field-specific interlaced locations are outside their contract.

For a decoder object that exposes a readiness wait and signalling snapshot, the connection has this shape:

The `"unpacked"` route below is an application-adapter result, not a catch-all decoder format. Select it only after
the adapter has proved that the surface is an ordinary device array with one of the layouts accepted by
`px.io.from_array`. Reject every other packed or unknown surface format.

```python
import cupy as cp
import pixtreme as px

consumer_stream = cp.cuda.Stream(non_blocking=True)
with consumer_stream:
    decoded.wait_on(consumer_stream.ptr)  # Backend-specific device-side readiness wait.
    if decoded.format == "nv12":
        frame = px.io.from_nv12(
            decoded.array,
            width=decoded.width,
            height=decoded.height,
            colorspace=signal.colorspace,
            gamma=signal.gamma,
            matrix=signal.matrix,
            range=signal.range,
            siting=signal.siting,
        )
    elif decoded.format == "p010":
        frame = px.io.from_p010(
            decoded.array,
            width=decoded.width,
            height=decoded.height,
            colorspace=signal.colorspace,
            gamma=signal.gamma,
            matrix=signal.matrix,
            range=signal.range,
            siting=signal.siting,
        )
    elif decoded.format == "unpacked":
        frame = px.io.from_array(
            decoded.array,
            colorspace=signal.colorspace,
            gamma=signal.gamma,
            channels=signal.channels,
            matrix=signal.matrix,
            layout=signal.layout,
        )
    else:
        raise ValueError(f"Unsupported decoder surface format: {decoded.format!r}")
```

The `decoded` and `signal` names represent backend-owned records; they are not pixtreme types. Validate that the CUDA
device, dimensions, dtype, packing, and reported surface format agree before calling a pixtreme boundary.

## Preserve matrix, range, and siting on the encoder path

`px.io.from_nv12` and `px.io.from_p010` expand the selected input range to full-range float and stamp `Frame.matrix`.
The inverse packed exits consume a full-range float32 Frame with exact `("Y", "Cb", "Cr")` channels. They quantize
with the requested `range=` and `siting=`, but they do not perform an RGB-to-YCbCr conversion or rematrix the Frame.

If processing produced RGB, explicitly create the encoder's YCbCr representation first. If processing retained YCbCr
but changed its colour representation, use `px.color.ycbcr_to_ycbcr` to select the output matrix. Then pack with the
same range and siting that the encoder will signal. The example supports exactly 8-bit NV12 and 10-bit P010; reject
other bit depths unless the application provides a separately documented packing boundary:

```python
producer_stream = cp.cuda.Stream(non_blocking=True)
with producer_stream:
    encoder_ycbcr = px.color.rgb_to_ycbcr(
        processed_rgb,
        colorspace=output_signal.colorspace,
        gamma=output_signal.gamma,
        matrix=output_signal.matrix,
    )
    if output_signal.bit_depth == 8:
        packed = px.io.to_nv12(
            encoder_ycbcr,
            range=output_signal.range,
            siting=output_signal.siting,
        )
    elif output_signal.bit_depth == 10:
        packed = px.io.to_p010(
            encoder_ycbcr,
            range=output_signal.range,
            siting=output_signal.siting,
        )
    else:
        raise ValueError(f"Unsupported encoder bit depth: {output_signal.bit_depth!r}")

encoder.submit(packed, producer_stream_handle=producer_stream.ptr)
```

The `encoder` and `output_signal` names represent application-owned records; they are not pixtreme types.
The packed array carries no colour metadata. Configure the encoder's VUI or container signal separately with the same
matrix, range, primaries, transfer, and chroma location. Matching function arguments alone cannot update encoded
bitstream signalling.

## CUDA stream and lifetime contract

The safe hand-off is stream ordered and asynchronous:

1. Select the current CuPy stream on which pixtreme should consume or produce the raw surface.
2. Before decode consumption, enqueue the decoder's readiness wait on that consumer stream. For a non-CuPy DLPack
   producer passed to `px.io.from_array`, CuPy passes the current stream as the DLPack consumer stream so the producer
   can enqueue its protocol-defined event/wait handshake.
3. Call the pixtreme boundary inside that stream context. Array repacking and NV12/P010 conversion kernels enqueue on
   the current CuPy stream. The call does not perform an implicit host synchronization.
4. Continue inference or image processing on the same stream, or record a CUDA event after pixtreme work and make each
   different consumer stream wait for it.
5. For encoding, give the encoder the producing stream handle or an equivalent CUDA event. If an external consumer has
   no stream-ordering interface, producing-stream synchronization is the explicit fallback before hand-off; device-wide
   synchronization is not the normal path.
6. Keep the decoder surface, DLPack producer, pixtreme Frame, packed array, and any external owner alive until every
   dependent GPU operation has completed.

Calling a conversion inside one stream and immediately consuming its result on an unrelated stream without an event
wait is a data race. Synchronizing only the producing stream is sufficient for a subsequent host read; it should not be
inserted between GPU stages that can exchange stream handles or events.

See [Tokens](tokens.md#matrix) for matrix and range semantics and [chroma siting](tokens.md#chroma-siting) for the
supported progressive 4:2:0 phases.
