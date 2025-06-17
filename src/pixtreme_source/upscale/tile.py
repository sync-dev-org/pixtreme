import cupy as cp
import torch

from ..utils.dlpack import to_cupy, to_tensor
from ..utils.dtypes import to_float32


def to_batch(tiles: list[cp.ndarray]) -> torch.Tensor:
    """
    Convert a list of tiles to a batch tensor.

    Parameters
    ----------
    tiles : list[cp.ndarray]
        List of image tiles in the shape (tile_size, tile_size, channel) in RGB format.

    Returns
    -------
    torch.Tensor
        Batch tensor in the shape (N, channel, tile_size, tile_size).
    """
    batch = []
    for i, tile in enumerate(tiles):
        tile = to_float32(tile)
        tile = to_tensor(tile)
        batch.append(tile)
    result = torch.cat(batch, dim=0)
    return result


def batch_to_tile(batch: torch.Tensor) -> list[cp.ndarray]:
    """
    Convert a batch tensor to a list of tiles.

    Parameters
    ----------
    batch : torch.Tensor
        Batch tensor in the shape (N, channel, tile_size, tile_size).

    Returns
    -------
    list[cp.ndarray]
        List of image tiles in the shape (tile_size, tile_size, channel) in RGB format.
    """
    tiles = []
    for i in range(batch.shape[0]):
        tile = to_cupy(batch[i])
        tiles.append(tile)
    return tiles


def tile_image(input_image: cp.ndarray, tile_size: int = 128, overlap: int = 16) -> tuple[list[cp.ndarray], tuple]:
    """
    Split the input image into overlapping tiles.

    Parameters
    ----------
    input_image : cp.ndarray
        Input image in the shape (height, width, channel) in RGB format.
    tile_size : int, optional
        Size of each tile, by default 128.
    overlap : int, optional
        Overlap between tiles, by default 16.

    Returns
    -------
    list[cp.ndarray]
        List of image tiles, each in the shape (tile_size, tile_size, channel) in RGB format.
    """
    input_image = add_padding(input_image, patch_size=tile_size, overlap=overlap)
    height, width, channels = input_image.shape
    tiles = []

    step = tile_size - overlap
    for y in range(0, height - tile_size + 1, step):
        for x in range(0, width - tile_size + 1, step):
            tile = input_image[y : y + tile_size, x : x + tile_size]
            tiles.append(tile)

    return tiles, input_image.shape


def add_padding(input_image: cp.ndarray, patch_size: int = 128, overlap: int = 16) -> cp.ndarray:
    pad_height = (patch_size - overlap) - (input_image.shape[0] - overlap) % (patch_size - overlap)
    pad_width = (patch_size - overlap) - (input_image.shape[1] - overlap) % (patch_size - overlap)
    if pad_height == patch_size - overlap:
        pad_height = 0
    if pad_width == patch_size - overlap:
        pad_width = 0

    padded_blob = cp.pad(input_image, ((0, pad_height), (0, pad_width), (0, 0)), mode="reflect")
    return padded_blob


def merge_tiles(
    tiles: list[cp.ndarray],
    original_shape: tuple[int, int, int],
    padded_shape: tuple[int, int, int],
    scale: int,
    tile_size: int = 128,
    overlap: int = 16,
) -> cp.ndarray:

    tile_size = tile_size * scale
    overlap = overlap * scale
    original_shape = (original_shape[0] * scale, original_shape[1] * scale, original_shape[2])

    padded_shape = (padded_shape[0] * scale, padded_shape[1] * scale, padded_shape[2])
    height, width, channels = padded_shape

    merged_image = cp.zeros(padded_shape, dtype=cp.float32)
    weights = cp.zeros(padded_shape, dtype=cp.float32)

    step = tile_size - overlap
    tile_index = 0

    # Create a Gaussian weight map for blending
    gaussian_weights = create_gaussian_weights(tile_size, sigma=tile_size // 4)

    for y in range(0, height - tile_size + 1, step):
        for x in range(0, width - tile_size + 1, step):
            tile = tiles[tile_index]
            tile = to_float32(tile)

            weighted_tile = tile * gaussian_weights

            merged_image[y : y + tile_size, x : x + tile_size] += weighted_tile
            weights[y : y + tile_size, x : x + tile_size] += gaussian_weights

            tile_index += 1

    # Normalize the merged image
    epsilon = 1e-8
    merged_image = merged_image / (weights + epsilon)

    # Crop by the original shape * scale
    merged_image = merged_image[: original_shape[0], : original_shape[1]]
    # logger.debug(f"Merged image shape: {merged_image.shape}")

    return merged_image


def create_gaussian_weights(size: int, sigma: int) -> cp.ndarray:
    """
    Create a Gaussian weight map for tile blending.

    Parameters
    ----------
    size : int
        Size of the weight map.
    sigma : int
        Standard deviation for the Gaussian distribution.

    Returns
    -------
    cp.ndarray
        Gaussian weight map in the shape (size, size, 1).
    """
    x, y = cp.meshgrid(cp.linspace(-1, 1, size), cp.linspace(-1, 1, size))
    d = cp.sqrt(x * x + y * y)
    gaussian = cp.exp(-(d**2 / (2.0 * sigma**2)))
    gaussian = gaussian / cp.sum(gaussian)
    return gaussian[:, :, cp.newaxis]
