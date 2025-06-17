import cupy as cp
import numpy as np
import pytest

import pixtreme as px


class TestFilters:
    """Test image filtering functions"""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image with known patterns"""
        # Create a test image with sharp edges and gradients
        image = cp.zeros((100, 100, 3), dtype=cp.float32)

        # Add a white square in the center
        image[25:75, 25:75, :] = 1.0

        # Add some noise
        noise = cp.random.normal(0, 0.05, image.shape).astype(cp.float32)
        image = cp.clip(image + noise, 0, 1)

        return image

    @pytest.fixture
    def noisy_image(self):
        """Create a noisy image for denoising tests"""
        # Create a gradient image
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        xx, yy = np.meshgrid(x, y)

        clean_image = np.stack([xx, yy, (xx + yy) / 2], axis=2).astype(np.float32)

        # Add significant noise
        noise = np.random.normal(0, 0.1, clean_image.shape).astype(np.float32)
        noisy = np.clip(clean_image + noise, 0, 1)

        return cp.asarray(noisy)

    def test_gaussian_blur_basic(self, sample_image):
        """Test basic Gaussian blur"""
        with px.Device(0):
            # Apply Gaussian blur with default parameters
            blurred = px.gaussian_blur(sample_image, kernel_size=5, sigma=1.0)

            assert blurred.shape == sample_image.shape

            # Blurred image should have less sharp edges
            # Calculate edge strength using simple gradient
            orig_edges = cp.abs(cp.diff(sample_image, axis=0)).mean()
            blur_edges = cp.abs(cp.diff(blurred, axis=0)).mean()
            assert blur_edges < orig_edges

    def test_gaussian_blur_different_kernels(self, sample_image):
        """Test Gaussian blur with different kernel sizes"""
        with px.Device(0):
            # Small kernel
            blur_small = px.gaussian_blur(sample_image, kernel_size=3, sigma=0.5)

            # Large kernel
            blur_large = px.gaussian_blur(sample_image, kernel_size=15, sigma=5.0)

            # Larger kernel should produce more blur
            orig_variance = cp.var(sample_image)
            small_variance = cp.var(blur_small)
            large_variance = cp.var(blur_large)

            assert small_variance < orig_variance
            assert large_variance < small_variance

    def test_gaussian_blur_sigma_effect(self, sample_image):
        """Test Gaussian blur with different sigma values"""
        with px.Device(0):
            # Same kernel size, different sigma
            blur_low_sigma = px.gaussian_blur(sample_image, kernel_size=9, sigma=1.0)
            blur_high_sigma = px.gaussian_blur(sample_image, kernel_size=9, sigma=3.0)

            # Higher sigma should produce more blur
            low_sigma_var = cp.var(blur_low_sigma)
            high_sigma_var = cp.var(blur_high_sigma)

            assert high_sigma_var < low_sigma_var

    def test_gaussian_blur_edge_handling(self, sample_image):
        """Test Gaussian blur edge handling"""
        with px.Device(0):
            # Apply blur and check edges
            blurred = px.gaussian_blur(sample_image, kernel_size=7, sigma=2.0)

            # Check that edges are properly handled (no artifacts)
            # Edge pixels should be valid values
            assert cp.all(blurred[0, :] >= 0) and cp.all(blurred[0, :] <= 1)
            assert cp.all(blurred[-1, :] >= 0) and cp.all(blurred[-1, :] <= 1)
            assert cp.all(blurred[:, 0] >= 0) and cp.all(blurred[:, 0] <= 1)
            assert cp.all(blurred[:, -1] >= 0) and cp.all(blurred[:, -1] <= 1)

    def test_gaussian_blur_custom_kernel(self, sample_image):
        """Test Gaussian blur with pre-computed kernel"""
        with px.Device(0):
            # Create a custom Gaussian kernel
            kernel_size = 5
            sigma = 1.5
            x = cp.arange(kernel_size) - kernel_size // 2
            kernel_1d = cp.exp(-0.5 * (x / sigma) ** 2)
            kernel_1d /= kernel_1d.sum()
            kernel = cp.outer(kernel_1d, kernel_1d).astype(cp.float32)

            # Apply blur with custom kernel
            blurred = px.gaussian_blur(sample_image, kernel_size=kernel_size, sigma=sigma, kernel=kernel)

            assert blurred.shape == sample_image.shape

    def test_gaussian_blur_performance(self, sample_image):
        """Test Gaussian blur performance with different image sizes"""
        with px.Device(0):
            # Test with different image sizes
            sizes = [50, 100, 200]
            times = []

            for size in sizes:
                # Resize image
                resized = px.resize(sample_image, dsize=(size, size))

                # Time the blur operation
                cp.cuda.Stream.null.synchronize()
                start = cp.cuda.Event()
                end = cp.cuda.Event()

                start.record()
                blurred = px.gaussian_blur(resized, kernel_size=9, sigma=2.0)
                end.record()
                end.synchronize()

                elapsed_time = cp.cuda.get_elapsed_time(start, end)
                times.append(elapsed_time)

            # Larger images should take more time (but not linearly due to GPU parallelism)
            assert times[1] > times[0]
            assert times[2] > times[1]

    def test_gaussian_blur_multichannel(self):
        """Test Gaussian blur on images with different channel counts"""
        with px.Device(0):
            # Test with 1, 3, and 4 channels
            for channels in [1, 3, 4]:
                image = cp.random.rand(100, 100, channels).astype(cp.float32)
                blurred = px.gaussian_blur(image, kernel_size=5, sigma=1.0)

                assert blurred.shape == image.shape
                assert blurred.dtype == image.dtype

    def test_gaussian_blur_preserves_range(self, sample_image):
        """Test that Gaussian blur preserves the value range"""
        with px.Device(0):
            # Apply blur
            blurred = px.gaussian_blur(sample_image, kernel_size=9, sigma=3.0)

            # Check that values are still in [0, 1] range
            assert cp.min(blurred) >= 0
            assert cp.max(blurred) <= 1

            # The mean should be approximately preserved
            assert cp.abs(cp.mean(blurred) - cp.mean(sample_image)) < 0.01
