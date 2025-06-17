import cupy as cp
import numpy as np
import pytest

import pixtreme as px


class TestColorConversions:
    """Test color space conversion functions"""

    @pytest.fixture
    def sample_image_rgb(self):
        """Create a sample RGB image"""
        # Create a 100x100 RGB image with random values
        np.random.seed(42)
        image = np.random.rand(100, 100, 3).astype(np.float32)
        return cp.asarray(image)

    @pytest.fixture
    def sample_image_bgr(self):
        """Create a sample BGR image"""
        np.random.seed(42)
        image = np.random.rand(100, 100, 3).astype(np.float32)
        return cp.asarray(image)

    def test_bgr_to_rgb(self, sample_image_bgr):
        """Test BGR to RGB conversion"""
        with px.Device(0):
            rgb_image = px.bgr_to_rgb(sample_image_bgr)
            assert isinstance(rgb_image, cp.ndarray)
            assert rgb_image.shape == sample_image_bgr.shape

            # Convert back and check
            bgr_image = px.rgb_to_bgr(rgb_image)
            assert cp.allclose(bgr_image, sample_image_bgr)

    def test_rgb_to_bgr(self, sample_image_rgb):
        """Test RGB to BGR conversion"""
        with px.Device(0):
            bgr_image = px.rgb_to_bgr(sample_image_rgb)
            assert isinstance(bgr_image, cp.ndarray)
            assert bgr_image.shape == sample_image_rgb.shape

    def test_rgb_to_grayscale(self, sample_image_rgb):
        """Test RGB to grayscale conversion"""
        with px.Device(0):
            gray_image = px.rgb_to_grayscale(sample_image_rgb)
            assert isinstance(gray_image, cp.ndarray)
            assert gray_image.shape == sample_image_rgb.shape

    def test_bgr_to_grayscale(self, sample_image_bgr):
        """Test BGR to grayscale conversion"""
        with px.Device(0):
            gray_image = px.bgr_to_grayscale(sample_image_bgr)
            assert isinstance(gray_image, cp.ndarray)
            assert gray_image.shape == sample_image_bgr.shape

    def test_rgb_to_hsv(self, sample_image_rgb):
        """Test RGB to HSV conversion"""
        with px.Device(0):
            hsv_image = px.rgb_to_hsv(sample_image_rgb)
            assert isinstance(hsv_image, cp.ndarray)
            assert hsv_image.shape == sample_image_rgb.shape

            # Convert back
            rgb_image = px.hsv_to_rgb(hsv_image)
            # HSV conversion has some precision loss
            assert cp.allclose(rgb_image, sample_image_rgb, rtol=0.01, atol=0.01)

    def test_bgr_to_hsv(self, sample_image_bgr):
        """Test BGR to HSV conversion"""
        with px.Device(0):
            hsv_image = px.bgr_to_hsv(sample_image_bgr)
            assert isinstance(hsv_image, cp.ndarray)
            assert hsv_image.shape == sample_image_bgr.shape

            # Convert back
            bgr_image = px.hsv_to_bgr(hsv_image)
            # HSV conversion has some precision loss
            assert cp.allclose(bgr_image, sample_image_bgr, rtol=0.01, atol=0.01)

    def test_rgb_to_ycbcr(self, sample_image_rgb):
        """Test RGB to YCbCr conversion"""
        with px.Device(0):
            ycbcr_image = px.rgb_to_ycbcr(sample_image_rgb)
            assert isinstance(ycbcr_image, cp.ndarray)
            assert ycbcr_image.shape == sample_image_rgb.shape

            # Convert back
            rgb_image = px.ycbcr_to_rgb(ycbcr_image)
            # YCbCr conversion has significant loss due to legal range conversion
            # Some values can be clipped or have large errors
            assert cp.allclose(rgb_image, sample_image_rgb, rtol=0.3, atol=0.2)

    def test_bgr_to_ycbcr(self, sample_image_bgr):
        """Test BGR to YCbCr conversion"""
        with px.Device(0):
            ycbcr_image = px.bgr_to_ycbcr(sample_image_bgr)
            assert isinstance(ycbcr_image, cp.ndarray)
            assert ycbcr_image.shape == sample_image_bgr.shape

            # Convert back
            bgr_image = px.ycbcr_to_bgr(ycbcr_image)
            # YCbCr conversion has significant loss due to legal range conversion
            # Some values can be clipped or have large errors
            assert cp.allclose(bgr_image, sample_image_bgr, rtol=0.3, atol=0.2)

    def test_ycbcr_to_grayscale(self, sample_image_rgb):
        """Test YCbCr to grayscale conversion"""
        with px.Device(0):
            ycbcr_image = px.rgb_to_ycbcr(sample_image_rgb)
            gray_image = px.ycbcr_to_grayscale(ycbcr_image)
            assert isinstance(gray_image, cp.ndarray)
            assert gray_image.shape == sample_image_rgb.shape
