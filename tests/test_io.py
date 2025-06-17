import os
import tempfile

import cupy as cp
import numpy as np
import pytest

import pixtreme as px


class TestIO:
    """Test I/O operations without external file dependencies"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_image_uint8(self):
        """Create a sample uint8 image"""
        np.random.seed(42)
        image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        return cp.asarray(image)

    @pytest.fixture
    def sample_image_float32(self):
        """Create a sample float32 image"""
        np.random.seed(42)
        image = np.random.rand(100, 100, 3).astype(np.float32)
        return cp.asarray(image)

    def test_imwrite_imread_uint8(self, temp_dir, sample_image_uint8):
        """Test writing and reading uint8 images"""
        with px.Device(0):
            # Test PNG format
            png_path = os.path.join(temp_dir, "test_uint8.png")
            success = px.imwrite(png_path, sample_image_uint8)
            assert success
            assert os.path.exists(png_path)

            # Read back and verify
            loaded_image = px.imread(png_path)
            assert isinstance(loaded_image, cp.ndarray)
            assert loaded_image.shape == sample_image_uint8.shape
            assert loaded_image.dtype == cp.uint8

            # Test JPEG format
            jpg_path = os.path.join(temp_dir, "test_uint8.jpg")
            success = px.imwrite(jpg_path, sample_image_uint8)
            assert success
            assert os.path.exists(jpg_path)

    def test_imwrite_imread_float32(self, temp_dir, sample_image_float32):
        """Test writing and reading float32 images"""
        with px.Device(0):
            # Convert to uint8 for standard formats
            uint8_image = px.to_uint8(sample_image_float32)

            png_path = os.path.join(temp_dir, "test_float32.png")
            success = px.imwrite(png_path, uint8_image)
            assert success

            # Read back and convert to float32
            loaded_image = px.imread(png_path)
            loaded_float = px.to_float32(loaded_image)
            assert isinstance(loaded_float, cp.ndarray)
            assert loaded_float.dtype == cp.float32
            assert loaded_float.max() <= 1.0
            assert loaded_float.min() >= 0.0

    def test_imwrite_different_formats(self, temp_dir, sample_image_uint8):
        """Test imwrite with different file formats"""
        with px.Device(0):
            # Test JPEG format
            jpg_path = os.path.join(temp_dir, "test_format.jpg")
            success = px.imwrite(jpg_path, sample_image_uint8)
            assert success
            assert os.path.exists(jpg_path)

            # Test PNG format
            png_path = os.path.join(temp_dir, "test_format.png")
            success = px.imwrite(png_path, sample_image_uint8)
            assert success
            assert os.path.exists(png_path)

            # Test BMP format
            bmp_path = os.path.join(temp_dir, "test_format.bmp")
            success = px.imwrite(bmp_path, sample_image_uint8)
            assert success
            assert os.path.exists(bmp_path)

    def test_imread_nonexistent_file(self):
        """Test imread with non-existent file"""
        with px.Device(0):
            # This should handle the error gracefully
            # The actual behavior depends on the implementation
            try:
                image = px.imread("nonexistent_file.png")
                # If it returns something, it should be None or raise an exception
                assert image is None or image.size == 0
            except Exception:
                # Exception is acceptable for non-existent file
                pass

    def test_imwrite_invalid_path(self, sample_image_uint8):
        """Test imwrite with invalid path"""
        with px.Device(0):
            # Try to write to a non-existent directory
            invalid_path = "/invalid/path/test.png"
            success = px.imwrite(invalid_path, sample_image_uint8)
            assert not success

    def test_imread_different_formats(self, temp_dir, sample_image_uint8):
        """Test imread with different file formats"""
        with px.Device(0):
            # Test different formats
            formats = [".png", ".jpg", ".bmp"]

            for fmt in formats:
                file_path = os.path.join(temp_dir, f"test{fmt}")
                px.imwrite(file_path, sample_image_uint8)

                # Read back
                loaded_image = px.imread(file_path)
                assert isinstance(loaded_image, cp.ndarray)
                assert loaded_image.shape[0] == sample_image_uint8.shape[0]
                assert loaded_image.shape[1] == sample_image_uint8.shape[1]
                # JPEG might have slight differences due to compression
                if fmt != ".jpg":
                    assert cp.allclose(loaded_image, sample_image_uint8)
