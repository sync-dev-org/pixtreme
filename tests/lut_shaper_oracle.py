"""Independent host model and deterministic OCIO fixture inputs for v1-lut-shaper.

No pixtreme imports: the oracle is derived from AC-11-10, not production output.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

CASES = tuple((edge, curve) for edge in (17, 33) for curve in ("log", "power", "weak"))
TOLERANCE_REASON = (
    "atol=2e-6, rtol=0: ten times the independent host model's observed 2e-7 error; "
    "about 17 float32 epsilons for two interpolation stages and CPU/GPU operation ordering."
)


def quantized_tables(edge: int, curve: str) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, edge)
    values = {
        "log": np.log1p(50 * axis) / np.log1p(50),
        "power": axis ** (1 / 2.2),
        "weak": axis**0.9,
    }[curve]
    rgb = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    luma = rgb @ np.asarray((0.2126, 0.7152, 0.0722))
    cube = np.clip(luma[..., None] + 1.4 * (rgb - luma[..., None]), 0, 1) ** 0.9
    return np.floor(values * 1023 + 0.5).astype(np.int64), np.floor(cube * 1023 + 0.5).astype(np.int64)


def three_dl_text(spacing: np.ndarray, cube: np.ndarray, *, lustre: bool = False) -> str:
    edge = len(spacing)
    header = f"3DMESH\nMesh {(edge - 1).bit_length() - 1} 10\n" if lustre else ""
    rows = [" ".join(str(int(code)) for code in spacing)]
    rows.extend(" ".join(str(int(code)) for code in row) for row in cube.reshape(-1, 3))
    return header + "\n".join(rows) + "\n"


def boundary_inputs(shaper: np.ndarray) -> np.ndarray:
    """AC-11-10's seven RGB tuples per knot/cell crossing, plus fixed seed points."""
    edge = len(shaper)
    knots = np.linspace(0.0, 1.0, edge)
    scalars = set(knots.tolist())
    for i, (lo, hi) in enumerate(zip(shaper[:-1], shaper[1:])):
        if lo == hi:
            continue
        for boundary in knots:
            fraction = (boundary - lo) / (hi - lo)
            if 0 <= fraction <= 1:
                scalars.add((i + fraction) / (edge - 1))
    points = []
    for scalar in sorted(scalars):
        points.append((scalar, scalar, scalar))
        for rgb in ((scalar, 1 - scalar, 0.5), (0.25, scalar, 0.75)):
            points.extend(rgb[shift:] + rgb[:shift] for shift in range(3))
    random = np.random.default_rng(11).uniform(0, 1, size=(20_000, 3))
    return np.unique(np.concatenate((random, points)).astype("<f4"), axis=0)


def host_cube(cube: np.ndarray, points: np.ndarray, interpolation: str = "tetrahedral") -> np.ndarray:
    """Float64 barycentric simplex oracle; trilinear uses independent corner weights."""
    cube = np.asarray(cube, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    position = np.clip(points, 0, 1) * (len(cube) - 1)
    lower = np.minimum(np.floor(position).astype(np.int64), len(cube) - 2)
    fractions = position - lower
    result = np.zeros(points.shape, dtype=np.float64)
    if interpolation == "trilinear":
        for corner in itertools.product((0, 1), repeat=3):
            indices = lower + corner
            weights = np.prod(np.where(corner, fractions, 1 - fractions), axis=-1)
            result += weights[..., None] * cube[tuple(indices.T)]
        return result
    order = np.argsort(-fractions, axis=-1, kind="stable")
    ranked = np.take_along_axis(fractions, order, axis=-1)
    weights = np.column_stack(
        (1 - ranked[:, 0], ranked[:, 0] - ranked[:, 1], ranked[:, 1] - ranked[:, 2], ranked[:, 2])
    )
    vertex = lower.copy()
    result += weights[:, :1] * cube[tuple(vertex.T)]
    for step in range(3):
        vertex[np.arange(len(vertex)), order[:, step]] += 1
        result += weights[:, step + 1, None] * cube[tuple(vertex.T)]
    return result


def host_apply(
    cube: np.ndarray,
    shaper: np.ndarray,
    points: np.ndarray,
    domain_min: tuple[float, ...] = (0.0, 0.0, 0.0),
    domain_max: tuple[float, ...] = (1.0, 1.0, 1.0),
    interpolation: str = "tetrahedral",
) -> np.ndarray:
    positions = np.clip((points.astype(np.float64) - domain_min) / (np.asarray(domain_max) - domain_min), 0, 1)
    shaped = np.interp(positions, np.linspace(0, 1, len(shaper)), shaper)
    return host_cube(cube, shaped, interpolation)


def float32_simplex_variants(vertices: np.ndarray, fractions: np.ndarray) -> dict[str, np.float32]:
    """Countermodels for AC-11-9 precision probes, not expected-output oracles.

    Enumerate weighted sums (sequential and paired) and edge differences, with
    every term ordering and with/without fused multiply-add. For these bounded
    float32 fixtures, float64 product-plus-sum followed by one float32 rounding
    models FMA; it does not round the product separately.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    fractions = np.asarray(fractions, dtype=np.float32)
    weights = np.asarray(
        (np.float32(1) - fractions[0], fractions[0] - fractions[1], fractions[1] - fractions[2], fractions[2]),
        dtype=np.float32,
    )

    def fma(a, b, c):
        return np.float32(np.float64(a) * np.float64(b) + np.float64(c))

    variants = {}
    products = vertices * weights
    for order in itertools.permutations(range(4)):
        separate = fused = products[order[0]]
        for j in order[1:]:
            separate = separate + products[j]
            fused = fma(vertices[j], weights[j], fused)
        variants[f"weighted {order}"] = separate
        variants[f"weighted-fma {order}"] = fused
        variants[f"weighted-pair {order}"] = (products[order[0]] + products[order[1]]) + (
            products[order[2]] + products[order[3]]
        )
    differences = np.diff(vertices)
    for order in itertools.permutations(range(3)):
        separate = fused = vertices[0]
        for j in order:
            separate = separate + fractions[j] * differences[j]
            fused = fma(fractions[j], differences[j], fused)
        variants[f"delta {order}"] = separate
        variants[f"delta-fma {order}"] = fused
    return variants


def fixture_bytes(edge: int, curve: str) -> dict[str, bytes]:
    """Generate from the pinned external CPU processor, with no GPU or production imports."""
    import PyOpenColorIO as ocio

    assert ocio.__version__ == "2.5.2", "regenerate only with the AC-11-11 pinned OCIO version"
    spacing, cube = quantized_tables(edge, curve)
    shaper = spacing.astype(np.float64) / 1023
    inputs = boundary_inputs(shaper)
    one = ocio.Lut1DTransform(length=edge, interpolation=ocio.INTERP_LINEAR)
    one.setData(np.repeat(shaper.astype(np.float32)[:, None], 3, axis=1))
    three = ocio.Lut3DTransform(gridSize=edge, interpolation=ocio.INTERP_TETRAHEDRAL)
    three.setData((cube.astype(np.float64) / 1023).astype(np.float32))
    processor = ocio.Config.CreateRaw().getProcessor(ocio.GroupTransform([one, three]))
    cpu = processor.getDefaultCPUProcessor()
    output = inputs.copy()
    cpu.applyRGB(output)
    metadata = {
        "array_order": "C",
        "atol": 2e-6,
        "curve": curve,
        "dtype": "<f4",
        "edge": edge,
        "input_shape": list(inputs.shape),
        "ocio_version": ocio.__version__,
        "processor": "raw config / forward linear shared Lut1D -> tetrahedral Lut3D / default CPU float32",
        "processor_cache_id": processor.getCacheID(),
        "cpu_cache_id": cpu.getCacheID(),
        "rtol": 0,
        "serialization_order": ["metadata.json", "source.3dl", "input.f32", "output.f32"],
        "tolerance_reason": TOLERANCE_REASON,
    }
    return {
        "metadata.json": (
            json.dumps(metadata, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode(),
        "source.3dl": three_dl_text(spacing, cube).encode("utf-8"),
        "input.f32": inputs.astype("<f4").tobytes(order="C"),
        "output.f32": output.astype("<f4").tobytes(order="C"),
    }


def load_fixture(root: Path, edge: int, curve: str) -> tuple[dict, bytes, np.ndarray, np.ndarray]:
    directory = root / f"{edge}-{curve}"
    metadata = json.loads((directory / "metadata.json").read_bytes())
    shape = tuple(metadata["input_shape"])
    inputs = np.frombuffer((directory / "input.f32").read_bytes(), dtype="<f4").reshape(shape)
    outputs = np.frombuffer((directory / "output.f32").read_bytes(), dtype="<f4").reshape(shape)
    return metadata, (directory / "source.3dl").read_bytes(), inputs, outputs
