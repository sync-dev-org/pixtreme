"""Literal channel routing across one or more Frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real

import cupy as cp
import numpy as np

from pixtreme._color.transform import rgb_to_rgb
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._core.value_domain import _float32_conversion_guidance

_RGB_CHANNELS = frozenset(("R", "G", "B"))
_YCBCR_CHANNELS = frozenset(("Y", "Cb", "Cr"))

_NormalizedSource = tuple[Frame, int] | np.float32
_NormalizedOutputs = tuple[tuple[str, _NormalizedSource], ...]


def _normalize_outputs(outputs: Mapping[str, object]) -> tuple[_NormalizedOutputs, Frame]:
    if not outputs:
        raise ValueError(
            _actionable_error(
                why="shuffle requires at least one output channel",
                what="received zero output channels",
                how="pass one or more output_label=(frame, source_label) or numeric fill keyword arguments",
            )
        )

    normalized: list[tuple[str, _NormalizedSource]] = []
    master: Frame | None = None
    for output_label, source in outputs.items():
        if not isinstance(output_label, str) or not output_label:
            raise ValueError(
                _actionable_error(
                    why="every shuffle output key must be one non-empty string label",
                    what=f"received output key {output_label!r}",
                    how="use one literal non-empty label per keyword or **dict key",
                )
            )

        if isinstance(source, tuple):
            if len(source) != 2:
                raise ValueError(
                    _actionable_error(
                        why="a shuffle Frame source must be a two-element tuple",
                        what=f"output {output_label!r} received tuple length {len(source)}",
                        how="pass (frame, source_label)",
                    )
                )
            frame, source_label = source
            if not isinstance(frame, Frame) or not isinstance(source_label, str) or not source_label:
                raise ValueError(
                    _actionable_error(
                        why="a shuffle source tuple must contain a Frame and a non-empty string label",
                        what=(
                            f"output {output_label!r} received "
                            f"({type(frame).__module__}.{type(frame).__qualname__}, {source_label!r})"
                        ),
                        how="pass (frame, source_label)",
                    )
                )
            if source_label not in frame.channels:
                raise ValueError(
                    _actionable_error(
                        why="shuffle could not find the requested source label",
                        what=f"output {output_label!r} requested {source_label!r} from Frame channels {frame.channels!r}",
                        how="choose a source label present in that Frame or correct the Frame channels metadata",
                    )
                )
            source_dtype = np.dtype(frame.dtype)
            if source_dtype != np.dtype(np.float32):
                raise ValueError(
                    _actionable_error(
                        why="shuffle requires float32 data for every source Frame",
                        what=f"output {output_label!r} received source dtype {source_dtype.name!r}",
                        how=_float32_conversion_guidance(source_dtype),
                    )
                )
            if master is None:
                master = frame
            normalized.append((output_label, (frame, frame.channels.index(source_label))))
            continue

        if isinstance(source, Real) and not isinstance(source, bool):
            normalized.append((output_label, np.float32(source)))
            continue

        raise ValueError(
            _actionable_error(
                why="a shuffle source must be a (Frame, label) tuple or numeric fill",
                what=f"output {output_label!r} received {source!r}",
                how="pass a two-element tuple, int, or float; bool fills are not accepted",
            )
        )

    if master is None:
        raise ValueError(
            _actionable_error(
                why="shuffle needs at least one Frame source to define geometry and metadata",
                what="received constants only",
                how="add one or more output_label=(frame, source_label) keyword arguments",
            )
        )
    return tuple(normalized), master


def _unique_source_frames(outputs: _NormalizedOutputs) -> tuple[Frame, ...]:
    sources: list[Frame] = []
    identities: set[int] = set()
    for _output_label, source in outputs:
        if not isinstance(source, tuple):
            continue
        frame, _channel_index = source
        identity = id(frame)
        if identity not in identities:
            identities.add(identity)
            sources.append(frame)
    return tuple(sources)


def _mismatch_error(
    *,
    field: str,
    master_value: object,
    source_value: object,
) -> ValueError:
    geometry = field in {"width", "height"}
    mode = "adapt cannot change" if geometry else "adapt=False requires matching"
    return ValueError(
        _actionable_error(
            why=f"shuffle {mode} {field} values",
            what=f"master {field}={master_value!r}, source {field}={source_value!r}",
            how=(
                "resize the source explicitly with px.transform.resize before shuffle"
                if geometry
                else "make metadata equal before shuffle or pass adapt=True"
            ),
        )
    )


def _prepare_sources(
    outputs: _NormalizedOutputs,
    *,
    master: Frame,
    adapt: bool,
) -> dict[int, Frame]:
    prepared: dict[int, Frame] = {}
    for source in _unique_source_frames(outputs):
        if source.width != master.width:
            raise _mismatch_error(
                field="width",
                master_value=master.width,
                source_value=source.width,
            )
        if source.height != master.height:
            raise _mismatch_error(
                field="height",
                master_value=master.height,
                source_value=source.height,
            )

        current = source
        if source.colorspace != master.colorspace or source.gamma != master.gamma:
            if not adapt:
                if source.colorspace != master.colorspace:
                    raise _mismatch_error(
                        field="colorspace",
                        master_value=master.colorspace,
                        source_value=source.colorspace,
                    )
                raise _mismatch_error(
                    field="gamma",
                    master_value=master.gamma,
                    source_value=source.gamma,
                )
            try:
                current = rgb_to_rgb(
                    source,
                    output_colorspace=master.colorspace,
                    output_gamma=master.gamma,
                )
            except ValueError as error:
                raise ValueError(
                    _actionable_error(
                        why="shuffle adapt=True source conversion was rejected by px.color.rgb_to_rgb",
                        what=str(error),
                        how="pre-adapt a Frame containing R, G, and B or make its colorspace and gamma match the master",
                    )
                ) from error
        prepared[id(source)] = current
    return prepared


def _resolve_output_matrix(outputs: _NormalizedOutputs) -> str | None:
    output_labels = frozenset(output_label for output_label, _source in outputs)
    has_rgb = bool(output_labels & _RGB_CHANNELS)
    has_ycbcr = bool(output_labels & _YCBCR_CHANNELS)
    if has_rgb or not has_ycbcr:
        return None

    claims = tuple(
        (output_label, source[0].matrix)
        for output_label, source in outputs
        if output_label in _YCBCR_CHANNELS and isinstance(source, tuple)
    )
    tokens = frozenset(matrix for _output_label, matrix in claims if matrix is not None)
    if len(tokens) > 1:
        raise ValueError(
            _actionable_error(
                why="shuffle received conflicting matrix provenance claims and never performs implicit rematrix",
                what=f"claims={claims!r}",
                how="rematrix explicitly with px.color.ycbcr_to_ycbcr before routing or use matching matrix claims",
            )
        )
    if not claims or any(matrix is None for _output_label, matrix in claims):
        return None
    return next(iter(tokens))


def _build_frame(
    outputs: _NormalizedOutputs,
    *,
    master: Frame,
    prepared: Mapping[int, Frame],
    matrix: str | None,
) -> Frame:
    output = cp.empty((master.height, master.width, len(outputs)), dtype=cp.float32)
    for output_index, (_output_label, source) in enumerate(outputs):
        if isinstance(source, tuple):
            frame, source_index = source
            output[..., output_index] = prepared[id(frame)].data[..., source_index]
        else:
            output[..., output_index].fill(source)

    return Frame(
        data=output,
        colorspace=master.colorspace,
        gamma=master.gamma,
        channels=tuple(output_label for output_label, _source in outputs),
        matrix=matrix,
    )


def _route_frame(frame: Frame, output_channels: Sequence[str]) -> Frame:
    """Route an exact channel multiset for internal operation composition."""
    remaining = list(enumerate(frame.channels))
    normalized: list[tuple[str, _NormalizedSource]] = []
    for output_label in output_channels:
        match = next((index for index, (_, label) in enumerate(remaining) if label == output_label), None)
        if match is None:
            raise ValueError(
                _actionable_error(
                    why="internal channel routing could not find the requested source label occurrence",
                    what=f"requested {tuple(output_channels)!r} from Frame channels {frame.channels!r}",
                    how="route only an equal channel-label multiset",
                )
            )
        source_index, _source_label = remaining.pop(match)
        normalized.append((output_label, (frame, source_index)))
    resolved = tuple(normalized)
    return _build_frame(
        resolved,
        master=frame,
        prepared={id(frame): frame},
        matrix=_resolve_output_matrix(resolved),
    )


def shuffle(*, adapt: bool = False, **outputs: tuple[Frame, str] | float) -> Frame:
    """Route source channels and numeric fills into a new float32 Frame.

    Each keyword is one literal output label whose value is ``(frame,
    source_label)`` or a non-bool real fill; keyword insertion order is output
    channel order. At least one Frame source is required. Every source Frame must
    contain the requested label, use float32 data, and have the same geometry as
    the first Frame source, which supplies width, height, colorspace, and gamma.

    With ``adapt=False``, all source colorspace and gamma metadata must match the
    first Frame. With ``adapt=True``, each mismatched source identity is converted
    once through :func:`px.color.rgb_to_rgb`; conversion therefore requires that
    source to contain R, G, and B. Routing then bit-copies source slices or writes
    float32 fills without clipping.

    Matrix provenance is cleared for RGB or mixed RGB/YCbCr output. YCbCr-only
    output retains one shared non-``None`` source claim, preserves ``native``
    literally, and clears provenance when any contributing claim is ``None``.
    Conflicting non-``None`` claims raise rather than performing implicit
    rematrix.

    Returns a new Frame backed by C-contiguous float32 data with the requested
    channels and the first Frame's geometry, colorspace, and gamma. Invalid options, labels,
    sources, dtypes, geometry, metadata, or matrix provenance raise
    :class:`ValueError` with actionable context.
    """
    if not isinstance(adapt, bool):
        raise ValueError(
            _actionable_error(
                why="shuffle adapt option accepts bool only",
                what=f"received adapt={adapt!r}",
                how="label 'adapt' is reserved; pass adapt=False or adapt=True and use a different label for output",
            )
        )

    normalized, master = _normalize_outputs(outputs)
    output_matrix = _resolve_output_matrix(normalized)
    prepared = _prepare_sources(normalized, master=master, adapt=adapt)
    return _build_frame(normalized, master=master, prepared=prepared, matrix=output_matrix)
