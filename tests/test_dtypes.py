import cupy as cp
import numpy as np
import pytest

import pixtreme as px


class TestDtypeConversions:
    """Test data type conversion functions"""

    @pytest.fixture
    def sample_float32(self):
        """Create a sample float32 image"""
        np.random.seed(42)
        image = np.random.rand(50, 50, 3).astype(np.float32)
        return cp.asarray(image)

    @pytest.fixture
    def sample_uint8(self):
        """Create a sample uint8 image"""
        np.random.seed(42)
        image = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
        return cp.asarray(image)

    @pytest.fixture
    def sample_uint16(self):
        """Create a sample uint16 image"""
        np.random.seed(42)
        image = (np.random.rand(50, 50, 3) * 65535).astype(np.uint16)
        return cp.asarray(image)

    @pytest.fixture
    def sample_float16(self):
        """Create a sample float16 image"""
        np.random.seed(42)
        image = np.random.rand(50, 50, 3).astype(np.float16)
        return cp.asarray(image)

    def test_to_uint8_from_float32(self, sample_float32):
        """Test float32 to uint8 conversion"""
        with px.Device(0):
            uint8_image = px.to_uint8(sample_float32)
            assert uint8_image.dtype == cp.uint8
            assert uint8_image.shape == sample_float32.shape
            assert uint8_image.max() <= 255
            assert uint8_image.min() >= 0

    def test_to_uint8_from_uint16(self, sample_uint16):
        """Test uint16 to uint8 conversion"""
        with px.Device(0):
            uint8_image = px.to_uint8(sample_uint16)
            assert uint8_image.dtype == cp.uint8
            assert uint8_image.shape == sample_uint16.shape

    def test_to_uint16_from_float32(self, sample_float32):
        """Test float32 to uint16 conversion"""
        with px.Device(0):
            uint16_image = px.to_uint16(sample_float32)
            assert uint16_image.dtype == cp.uint16
            assert uint16_image.shape == sample_float32.shape
            assert uint16_image.max() <= 65535
            assert uint16_image.min() >= 0

    def test_to_uint16_from_uint8(self, sample_uint8):
        """Test uint8 to uint16 conversion"""
        with px.Device(0):
            uint16_image = px.to_uint16(sample_uint8)
            assert uint16_image.dtype == cp.uint16
            assert uint16_image.shape == sample_uint8.shape

    def test_to_float16_from_uint8(self, sample_uint8):
        """Test uint8 to float16 conversion"""
        with px.Device(0):
            float16_image = px.to_float16(sample_uint8)
            assert float16_image.dtype == cp.float16
            assert float16_image.shape == sample_uint8.shape
            assert float16_image.max() <= 1.0
            assert float16_image.min() >= 0.0

    def test_to_float32_from_uint8(self, sample_uint8):
        """Test uint8 to float32 conversion"""
        with px.Device(0):
            float32_image = px.to_float32(sample_uint8)
            assert float32_image.dtype == cp.float32
            assert float32_image.shape == sample_uint8.shape
            assert float32_image.max() <= 1.0
            assert float32_image.min() >= 0.0

    def test_to_float32_from_uint16(self, sample_uint16):
        """Test uint16 to float32 conversion"""
        with px.Device(0):
            float32_image = px.to_float32(sample_uint16)
            assert float32_image.dtype == cp.float32
            assert float32_image.shape == sample_uint16.shape
            assert float32_image.max() <= 1.0
            assert float32_image.min() >= 0.0

    def test_to_dtype_uint8(self, sample_float32):
        """Test generic to_dtype function for uint8"""
        with px.Device(0):
            uint8_image = px.to_dtype(sample_float32, "uint8")
            assert uint8_image.dtype == cp.uint8
            assert uint8_image.shape == sample_float32.shape

    def test_to_dtype_float32(self, sample_uint8):
        """Test generic to_dtype function for float32"""
        with px.Device(0):
            float32_image = px.to_dtype(sample_uint8, "float32")
            assert float32_image.dtype == cp.float32
            assert float32_image.shape == sample_uint8.shape

    def test_round_trip_conversions(self):
        """Test round-trip conversions maintain data integrity"""
        with px.Device(0):
            # Create original data
            np.random.seed(42)
            original = cp.asarray(np.random.rand(10, 10, 3).astype(np.float32))

            # Float32 -> uint8 -> float32
            uint8_temp = px.to_uint8(original)
            recovered = px.to_float32(uint8_temp)
            # Check that values are close (accounting for quantization)
            assert cp.allclose(original, recovered, atol=1.0 / 255)

            # Float32 -> uint16 -> float32
            uint16_temp = px.to_uint16(original)
            recovered = px.to_float32(uint16_temp)
            # Much closer due to higher precision
            assert cp.allclose(original, recovered, atol=1.0 / 65535)
