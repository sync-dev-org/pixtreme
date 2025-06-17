import cupy as cp
import numpy as np


def to_uint8(image: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
    if image.dtype == "uint8":
        pass
    elif image.dtype == "float32":
        image = (image.astype("float32") * 255.0).clip(0, 255.0).astype("uint8")
    elif image.dtype == "uint16":
        image = (image.astype("float32") / 65535.0 * 255.0).clip(0, 255.0).astype("uint8")
    elif image.dtype == "float16":
        image = (image.astype("float32") * 255.0).clip(0, 255).astype("uint8")
    elif image.dtype == "float64":
        image = (image.astype("float64") * 255.0).clip(0, 255).astype("uint8")
    else:
        raise ValueError(f"Unsupported dtype {image.dtype}")
    return image


def to_uint16(image: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
    if image.dtype == "uint16":
        pass
    elif image.dtype == "float32":
        image = (image.astype("float32") * 65535.0).clip(0, 65535).astype("uint16")
    elif image.dtype == "uint8":
        image = (image.astype("float32") / 255 * 65535).clip(0, 65535).astype("uint16")
    elif image.dtype == "float16":
        image = (image.astype("float32") * 65535.0).clip(0, 65535).astype("uint16")
    elif image.dtype == "float64":
        image = (image.astype("float64") * 65535.0).clip(0, 65535).astype("uint16")
    else:
        raise ValueError(f"Unsupported dtype {image.dtype}")
    return image


def to_float16(image: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
    if image.dtype == "float16":
        pass
    elif image.dtype == "float32":
        image = image.astype("float16")
    elif image.dtype == "uint8":
        image = (image.astype("float32") / 255.0).astype("float16")
    elif image.dtype == "uint16":
        image = (image.astype("float32") / 65535.0).astype("float16")
    elif image.dtype == "float64":
        image = image.astype("float16")
    else:
        raise ValueError(f"Unsupported dtype {image.dtype}")
    return image


def to_float32(image: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
    if image.dtype == "float32":
        pass
    elif image.dtype == "uint8":
        image = image.astype("float32") / 255.0
    elif image.dtype == "uint16":
        image = image.astype("float32") / 65535.0
    elif image.dtype == "float16":
        image = image.astype("float32")
    elif image.dtype == "float64":
        image = image.astype("float32")
    else:
        raise ValueError(f"Unsupported dtype {image.dtype}")
    image = image.clip(0, 1.0)
    return image


def to_float64(image: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
    if image.dtype == "float64":
        pass
    elif image.dtype == "float32":
        image = image.astype("float64")
    elif image.dtype == "uint8":
        image = (image.astype("float32") / 255.0).astype("float64")
    elif image.dtype == "uint16":
        image = (image.astype("float32") / 65535.0).astype("float64")
    elif image.dtype == "float16":
        image = image.astype("float64")
    else:
        raise ValueError(f"Unsupported dtype {image.dtype}")
    return image


def to_dtype(image: np.ndarray | cp.ndarray, dtype: str) -> np.ndarray | cp.ndarray:
    if dtype == "uint8":
        image = to_uint8(image)
    elif dtype == "uint16":
        image = to_uint16(image)
    elif dtype == "float16":
        image = to_float16(image)
    elif dtype == "float32":
        image = to_float32(image)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")
    return image
