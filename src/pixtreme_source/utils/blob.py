import cupy as cp

from ..color.bgr import bgr_to_rgb
from ..transform.resize import resize


def to_blobs(
    images: list[cp.ndarray],
    scalefactor: float = 1.0,
    size=None,
    mean=(0, 0, 0),
    swapRB: bool = False,
    fp16: bool = False,
) -> cp.ndarray:
    """
    Convert a list of images to blobs.
    """
    blobs = []
    for image in images:
        blobs.append(to_blob(image, scalefactor, size, mean, swapRB, fp16))

    # list[Union[np.ndarray]] -> np.ndarray shape (N, C, H, W)
    blobs = cp.concatenate(blobs, axis=0)

    return blobs


def to_blob(
    image: cp.ndarray,
    scalefactor: float = 1.0,
    size=None,
    mean=(0, 0, 0),
    swapRB: bool = False,
    fp16: bool = False,
) -> cp.ndarray:
    """
    Convert an image to a blob.

    Parameters
    ----------
    image : cp.ndarray
        The input image in RGB format.
    scalefactor : float (optional)
        The scale factor to apply. by default 1.0
    size : tuple (optional)
        The size of the output image. by default None
    mean : tuple (optional)
        The mean value to subtract. by default (0, 0, 0)
    swapRB : bool (optional)
        Swap the R and B channels. by default False

    Returns
    -------
    cp.ndarray
        The blob image in CHW format.
    """
    array_dtype = image.dtype

    if size is not None:
        image = resize(image, (size[1], size[0]))

    if swapRB:
        image = bgr_to_rgb(image)

    image -= cp.array(mean, dtype=array_dtype)
    image *= scalefactor

    image = image.transpose(2, 0, 1)  # HWC to CHW
    image = image[cp.newaxis, :, :, :].astype(array_dtype)  # Add batch dimension

    return image
