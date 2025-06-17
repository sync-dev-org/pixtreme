import cupy as cp
import numpy as np
import pytest

import pixtreme as px


@pytest.fixture
def sample_image():
    """Create a sample RGB image"""
    # Create a 512x512 RGB image with random values
    np.random.seed(42)
    image = cp.random.rand(512, 512, 3).astype(cp.float32)
    # image_path = "examples/example.png"
    # image = px.imread(image_path)
    image = px.to_float32(image)
    return image


def test_bgr_to_rgb(sample_image):
    """Test BGR to RGB conversion"""
    with px.Device(0):
        rgb_image = px.bgr_to_rgb(sample_image)
        assert isinstance(rgb_image, cp.ndarray)
        assert rgb_image.shape == sample_image.shape

        # Convert back and check
        bgr_image = px.rgb_to_bgr(rgb_image)
        assert cp.allclose(bgr_image, sample_image)


def test_rgb_to_bgr(sample_image):
    """Test RGB to BGR conversion"""
    with px.Device(0):
        bgr_image = px.rgb_to_bgr(sample_image)
        assert isinstance(bgr_image, cp.ndarray)
        assert bgr_image.shape == sample_image.shape

        # Convert back and check
        rgb_image = px.bgr_to_rgb(bgr_image)
        assert cp.allclose(rgb_image, sample_image)


def test_rgb_to_grayscale(sample_image):
    """Test RGB to grayscale conversion"""
    with px.Device(0):
        gray_image = px.rgb_to_grayscale(sample_image)
        assert isinstance(gray_image, cp.ndarray)
        assert gray_image.shape == sample_image.shape


def test_bgr_to_grayscale(sample_image):
    """Test BGR to grayscale conversion"""
    with px.Device(0):
        gray_image = px.bgr_to_grayscale(sample_image)
        assert isinstance(gray_image, cp.ndarray)
        assert gray_image.shape == sample_image.shape


def test_rgb_to_hsv(sample_image):
    """Test RGB to HSV conversion"""
    with px.Device(0):
        input_rgb_image = px.bgr_to_rgb(sample_image)
        hsv_image = px.rgb_to_hsv(input_rgb_image)
        assert isinstance(hsv_image, cp.ndarray)
        assert hsv_image.shape == sample_image.shape

        # Convert back
        output_rgb_image = px.hsv_to_rgb(hsv_image)
        output_bgr_image = px.rgb_to_bgr(output_rgb_image)
        # HSV conversion has some precision loss
        assert cp.allclose(output_bgr_image, sample_image, rtol=0.01, atol=0.01), (
            f"output_bgr_image and sample_image are not close enough: {cp.max(cp.abs(output_bgr_image - sample_image))}"
        )


def test_bgr_to_hsv(sample_image):
    """Test BGR to HSV conversion"""
    with px.Device(0):
        hsv_image = px.bgr_to_hsv(sample_image)
        assert isinstance(hsv_image, cp.ndarray)
        assert hsv_image.shape == sample_image.shape

        # Convert back
        bgr_image = px.hsv_to_bgr(hsv_image)
        # HSV conversion has some precision loss
        assert cp.allclose(bgr_image, sample_image, rtol=0.01, atol=0.01), (
            f"bgr_image and sample_image are not close enough: {cp.max(cp.abs(bgr_image - sample_image))}"
        )


def test_rgb_to_ycbcr(sample_image):
    """Test RGB to YCbCr conversion"""
    with px.Device(0):
        px.imshow("Sample Image", sample_image)

        input_rgb_image = px.bgr_to_rgb(sample_image)

        ycbcr_image = px.rgb_to_ycbcr(input_rgb_image)
        assert isinstance(ycbcr_image, cp.ndarray)
        assert ycbcr_image.shape == sample_image.shape

        # Convert back
        output_rgb_image = px.ycbcr_to_rgb(ycbcr_image)
        output_bgr_image = px.rgb_to_bgr(output_rgb_image)
        # YCbCr conversion has significant loss due to legal range conversion
        # Some values can be clipped or have large errors
        assert cp.allclose(output_bgr_image, sample_image, rtol=0.01, atol=0.01), (
            f"output_bgr_image and sample_image are not close enough: {cp.max(cp.abs(output_bgr_image - sample_image))}"
        )

        px.imshow("YCbCr Image", ycbcr_image)
        px.imshow("Output BGR Image", output_bgr_image)
        px.waitkey(0)
        px.destroy_all_windows()


def test_bgr_to_ycbcr(sample_image):
    """Test BGR to YCbCr conversion"""
    with px.Device(0):
        ycbcr_image = px.bgr_to_ycbcr(sample_image)
        assert isinstance(ycbcr_image, cp.ndarray)
        assert ycbcr_image.shape == sample_image.shape

        # Convert back
        bgr_image = px.ycbcr_to_bgr(ycbcr_image)
        # YCbCr conversion has significant loss due to legal range conversion
        # Some values can be clipped or have large errors
        assert cp.allclose(bgr_image, sample_image, rtol=0.3, atol=0.2)


def test_ycbcr_to_grayscale(sample_image):
    """Test YCbCr to grayscale conversion"""
    with px.Device(0):
        ycbcr_image = px.rgb_to_ycbcr(sample_image)
        gray_image = px.ycbcr_to_grayscale(ycbcr_image)
        assert isinstance(gray_image, cp.ndarray)
        assert gray_image.shape == sample_image.shape


def test_rec709_to_aces2065_1(sample_image):
    """Test Rec.709 to ACES2065-1 conversion"""
    with px.Device(0):
        aces_image = px.rec709_to_aces2065_1(sample_image)
        assert isinstance(aces_image, cp.ndarray)
        assert aces_image.shape == sample_image.shape

        # Convert back
        rec709_image = px.aces2065_1_to_rec709(aces_image)

        # Check if the conversion is close enough
        assert cp.allclose(rec709_image, sample_image, rtol=0.6, atol=0.6), (
            f"rec709_image and sample_image are not close enough: {cp.max(cp.abs(rec709_image - sample_image))}"
        )


def test_aces2065_1_to_rec709(sample_image):
    """Test ACES2065-1 to Rec.709 conversion"""
    with px.Device(0):
        aces_image = px.rec709_to_aces2065_1(sample_image)
        rec709_image = px.aces2065_1_to_rec709(aces_image)
        assert isinstance(rec709_image, cp.ndarray)
        assert rec709_image.shape == sample_image.shape

        # Convert back
        aces_back = px.rec709_to_aces2065_1(rec709_image)
        assert cp.allclose(aces_back, aces_image, rtol=0.6, atol=0.6), (
            f"aces_back and aces_image are not close enough: {cp.max(cp.abs(aces_back - aces_image))}"
        )


def test_aces2065_1_to_acescct(sample_image):
    """Test ACES2065-1 to ACESCCT conversion"""
    with px.Device(0):
        acescct_image = px.aces2065_1_to_acescct(sample_image)
        assert isinstance(acescct_image, cp.ndarray)
        assert acescct_image.shape == sample_image.shape

        # Convert back
        aces_back = px.acescct_to_aces2065_1(acescct_image)
        assert cp.allclose(aces_back, sample_image, rtol=0.6, atol=0.6), (
            f"aces_back and sample_image are not close enough: {cp.max(cp.abs(aces_back - sample_image))}"
        )


def test_acescct_to_aces2065_1(sample_image):
    """Test ACESCCT to ACES2065-1 conversion"""
    with px.Device(0):
        acescct_image = px.aces2065_1_to_acescct(sample_image)
        aces_image = px.acescct_to_aces2065_1(acescct_image)
        assert isinstance(aces_image, cp.ndarray)
        assert aces_image.shape == sample_image.shape

        # Convert back
        acescct_back = px.aces2065_1_to_acescct(aces_image)
        assert cp.allclose(acescct_back, acescct_image, rtol=0.6, atol=0.6), (
            f"acescct_back and acescct_image are not close enough: {cp.max(cp.abs(acescct_back - acescct_image))}"
        )


def test_aces2065_1_to_acescg(sample_image):
    """Test ACES2065-1 to ACEScg conversion"""
    with px.Device(0):
        acescg_image = px.aces2065_1_to_acescg(sample_image)
        assert isinstance(acescg_image, cp.ndarray)
        assert acescg_image.shape == sample_image.shape

        # Convert back
        aces_back = px.acescg_to_aces2065_1(acescg_image)
        assert cp.allclose(aces_back, sample_image, rtol=0.6, atol=0.6), (
            f"aces_back and sample_image are not close enough: {cp.max(cp.abs(aces_back - sample_image))}"
        )


def test_acescg_to_aces2065_1(sample_image):
    """Test ACEScg to ACES2065-1 conversion"""
    with px.Device(0):
        aces_image = px.acescg_to_aces2065_1(sample_image)
        assert isinstance(aces_image, cp.ndarray)
        assert aces_image.shape == sample_image.shape

        # Convert back
        acescg_back = px.aces2065_1_to_acescg(aces_image)
        assert cp.allclose(acescg_back, sample_image, rtol=0.6, atol=0.6), (
            f"acescg_back and sample_image are not close enough: {cp.max(cp.abs(acescg_back - sample_image))}"
        )
