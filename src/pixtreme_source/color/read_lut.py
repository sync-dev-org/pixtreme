import hashlib
import os

import cupy as cp


def read_lut(file_path: str, use_cache: bool = True, cache_dir="cache") -> cp.ndarray:
    """
    Read a 3D LUT Cube file and return the LUT data as a CuPy ndarray.

    Parameters
    ----------
    file_path : str
        The path to the LUT file. Must be a .cube file.
    use_cache : bool, optional
        Whether to use the cache, by default True
    cache_dir : str, optional
        The directory to store the cache, by default "cache"

    Returns
    -------
    lut_data : cp.ndarray
        The LUT data. The shape is (N, N, N, 3). dtype is float32.

    """
    try:
        if os.path.exists(file_path) is False:
            raise FileNotFoundError(f"Error: {file_path} not found.")

        os.makedirs(cache_dir, exist_ok=True)
        with open(file_path, "rb") as file:
            file_hash = hashlib.md5(file.read()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{file_hash}.npy")

        if use_cache and os.path.exists(cache_path):
            try:
                lut_data = cp.load(cache_path)
                return lut_data
            except Exception as e:
                os.remove(cache_path)
                print(f"Error read_lut: {e}")
                print(f"Rreading {file_path} again.")
                pass

        with open(file_path, "r") as file:
            lines = file.readlines()

        # Find the size of the LUT
        size_line = [line for line in lines if line.startswith("LUT_3D_SIZE")][0]
        size = int(size_line.split()[1])

        # Initialize the LUT data array
        lut_data = cp.zeros((size, size, size, 3), dtype=cp.float32)

        # Read the LUT data
        data_lines = [line for line in lines if len(line.split()) == 3]
        index = 0
        for line in data_lines:
            r, g, b = map(float, line.split())
            x = index % size
            y = (index // size) % size
            z = index // (size**2)
            lut_data[x, y, z] = cp.array([r, g, b], dtype=cp.float32)
            index += 1

        cp.save(cache_path, lut_data)
        return lut_data

    except Exception as e:
        print(f"Error read_lut: {e}")
        raise e
