"""Shared blend token vocabulary and CUDA blend-value source."""

from pixtreme._core.vocabulary import _BLEND_TOKENS as _BLEND_TOKENS

_BLEND_CODES = {token: index for index, token in enumerate(_BLEND_TOKENS)}
_DRAW_BLEND_TOKENS = tuple(_BLEND_TOKENS[index] for index in (0, 2, 5, 3))
_DRAW_BLEND_CODES = {token: _BLEND_CODES[token] for token in _DRAW_BLEND_TOKENS}

_BLEND_DEVICE_SOURCE = r"""
__device__ float pixtreme_blend(const float background, const float source, const int blend) {
    if (blend == 0) {
        return source;
    }
    if (blend == 1) {
        return fmaxf(background, source);
    }
    if (blend == 2) {
        return background + source;
    }
    if (blend == 3) {
        return 1.0f - (1.0f - background) * (1.0f - source);
    }
    if (blend == 4) {
        return fminf(background, source);
    }
    if (blend == 5) {
        return background * source;
    }
    if (blend == 6) {
        return fabsf(background - source);
    }
    if (blend == 7) {
        return background <= 0.5f
            ? 2.0f * background * source
            : 1.0f - 2.0f * (1.0f - background) * (1.0f - source);
    }
    if (blend == 8) {
        return source <= 0.5f
            ? 2.0f * background * source
            : 1.0f - 2.0f * (1.0f - background) * (1.0f - source);
    }
    if (source <= 0.5f) {
        return background - (1.0f - 2.0f * source) * background * (1.0f - background);
    }
    const float d = background <= 0.25f
        ? ((16.0f * background - 12.0f) * background + 4.0f) * background
        : sqrtf(background);
    return background + (2.0f * source - 1.0f) * (d - background);
}
"""
