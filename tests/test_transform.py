import cupy as cp
import numpy as np
import pytest

import pixtreme as px


class TestTransform:
    """Test image transformation functions"""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image with known patterns"""
        # Create a gradient image for testing transformations
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        xx, yy = np.meshgrid(x, y)

        # Create RGB channels with different patterns
        r_channel = xx
        g_channel = yy
        b_channel = (xx + yy) / 2

        image = np.stack([r_channel, g_channel, b_channel], axis=2).astype(np.float32)
        return cp.asarray(image)

    @pytest.fixture
    def identity_matrix(self):
        """Create an identity transformation matrix"""
        return cp.eye(2, 3, dtype=cp.float32)

    def test_resize_with_scale_factors(self, sample_image):
        """Test resize using scale factors instead of absolute size"""
        with px.Device(0):
            # Test downscaling by half
            resized = px.resize(sample_image, fx=0.5, fy=0.5)
            assert resized.shape[0] == sample_image.shape[0] // 2
            assert resized.shape[1] == sample_image.shape[1] // 2
            assert resized.shape[2] == sample_image.shape[2]

            # Test upscaling by 2x
            resized = px.resize(sample_image, fx=2.0, fy=2.0)
            assert resized.shape[0] == sample_image.shape[0] * 2
            assert resized.shape[1] == sample_image.shape[1] * 2

            # Test non-uniform scaling
            resized = px.resize(sample_image, fx=1.5, fy=0.75)
            assert resized.shape[0] == int(sample_image.shape[0] * 0.75)
            assert resized.shape[1] == int(sample_image.shape[1] * 1.5)

    def test_resize_auto_interpolation(self, sample_image):
        """Test INTER_AUTO interpolation selection"""
        with px.Device(0):
            # INTER_AUTO should select appropriate method based on scaling
            # Downscaling should use INTER_AREA
            small = px.resize(sample_image, dsize=(50, 50), interpolation=px.INTER_AUTO)
            assert small.shape == (50, 50, 3)

            # Upscaling should use INTER_CUBIC or similar
            large = px.resize(sample_image, dsize=(200, 200), interpolation=px.INTER_AUTO)
            assert large.shape == (200, 200, 3)

    def test_resize_edge_cases(self, sample_image):
        """Test resize with edge cases"""
        with px.Device(0):
            # Test 1x1 resize
            tiny = px.resize(sample_image, dsize=(1, 1))
            assert tiny.shape == (1, 1, 3)

            # Test same size (should return a copy)
            same = px.resize(sample_image, dsize=(100, 100))
            assert same.shape == sample_image.shape
            assert not cp.shares_memory(same, sample_image)

    def test_affine_transform_identity(self, sample_image, identity_matrix):
        """Test affine transform with identity matrix"""
        with px.Device(0):
            transformed = px.affine_transform(sample_image, identity_matrix, dsize=(100, 100))
            assert transformed.shape == sample_image.shape
            # Identity transform should preserve the image
            assert cp.allclose(transformed, sample_image, rtol=1e-5)

    def test_affine_transform_translation(self, sample_image):
        """Test affine transform with translation"""
        with px.Device(0):
            # Create translation matrix (shift by 10 pixels in x and y)
            matrix = cp.array([[1, 0, 10], [0, 1, 10]], dtype=cp.float32)

            transformed = px.affine_transform(sample_image, matrix, dsize=(100, 100))
            assert transformed.shape == sample_image.shape

            # Check that the image has been shifted
            # The top-left corner should now be black (or border value)
            assert cp.mean(transformed[:10, :10]) < cp.mean(sample_image[:10, :10])

    def test_affine_transform_rotation(self, sample_image):
        """Test affine transform with rotation"""
        with px.Device(0):
            # Create 90-degree rotation matrix around center
            angle = np.pi / 2  # 90 degrees
            cx, cy = 50, 50  # Center of 100x100 image

            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # Rotation matrix with center translation
            matrix = cp.array(
                [[cos_a, -sin_a, cx - cos_a * cx + sin_a * cy], [sin_a, cos_a, cy - sin_a * cx - cos_a * cy]],
                dtype=cp.float32,
            )

            transformed = px.affine_transform(sample_image, matrix, dsize=(100, 100))
            assert transformed.shape == sample_image.shape

    def test_affine_transform_scale(self, sample_image):
        """Test affine transform with scaling"""
        with px.Device(0):
            # Create scaling matrix (2x scale)
            matrix = cp.array([[2, 0, 0], [0, 2, 0]], dtype=cp.float32)

            # Output size should be same, but content scaled
            transformed = px.affine_transform(sample_image, matrix, dsize=(100, 100))
            assert transformed.shape == (100, 100, 3)

    def test_get_inverse_matrix(self):
        """Test inverse matrix calculation"""
        with px.Device(0):
            # Test with a simple transformation matrix
            matrix = cp.array([[2, 0, 10], [0, 2, 20]], dtype=cp.float32)

            inverse = px.get_inverse_matrix(matrix)

            # Verify that matrix @ inverse ≈ identity
            # Convert to 3x3 for multiplication
            mat3x3 = cp.vstack([matrix, cp.array([0, 0, 1])])
            inv3x3 = cp.vstack([inverse, cp.array([0, 0, 1])])

            result = cp.dot(mat3x3, inv3x3)
            expected = cp.eye(3)

            assert cp.allclose(result, expected, rtol=1e-5)

    def test_erode_basic(self, sample_image):
        """Test basic erosion operation"""
        with px.Device(0):
            # Convert to binary image for clearer erosion effect
            binary = (sample_image > 0.5).astype(cp.float32)

            # Apply erosion with default 3x3 kernel
            eroded = px.erode(binary, kernel_size=3)

            assert eroded.shape == binary.shape
            # Erosion should reduce the number of white pixels
            assert cp.sum(eroded) <= cp.sum(binary)

    def test_erode_with_iterations(self, sample_image):
        """Test erosion with multiple iterations"""
        with px.Device(0):
            binary = (sample_image > 0.5).astype(cp.float32)

            # Apply erosion multiple times to simulate iterations
            eroded1 = px.erode(binary, kernel_size=3)
            eroded3 = binary
            for _ in range(3):
                eroded3 = px.erode(eroded3, kernel_size=3)

            # More iterations should result in more erosion
            assert cp.sum(eroded3) <= cp.sum(eroded1)

    def test_erode_with_custom_kernel(self, sample_image):
        """Test erosion with custom kernel"""
        with px.Device(0):
            binary = (sample_image > 0.5).astype(cp.float32)

            # Create a custom 5x5 kernel
            kernel = cp.ones((5, 5), dtype=cp.int32)

            # Apply erosion with custom kernel
            eroded = px.erode(binary, kernel_size=5, kernel=kernel)

            assert eroded.shape == binary.shape
            # Larger kernel should cause more erosion than default 3x3
            eroded_default = px.erode(binary, kernel_size=3)
            assert cp.sum(eroded) <= cp.sum(eroded_default)

    def test_crop_from_kps(self):
        """Test cropping based on keypoints"""
        with px.Device(0):
            # Create a larger test image
            image = cp.random.rand(200, 200, 3).astype(cp.float32)

            # Define keypoints (e.g., face landmarks)
            # Using 5 points: left eye, right eye, nose, left mouth, right mouth
            kps = cp.array(
                [
                    [60, 60],  # left eye
                    [140, 60],  # right eye
                    [100, 100],  # nose
                    [70, 140],  # left mouth
                    [130, 140],  # right mouth
                ],
                dtype=cp.float32,
            )

            # Crop around keypoints
            cropped, transform_matrix = px.crop_from_kps(image, kps, size=112)

            assert cropped.shape == (112, 112, 3)
            assert transform_matrix.shape == (2, 3)

            # The cropped image should be a valid subregion
            assert cp.all(cropped >= 0) and cp.all(cropped <= 1)

    def test_resize_list_input(self, sample_image):
        """Test resize with list of images"""
        with px.Device(0):
            # Create a list of images
            images = [sample_image, sample_image * 0.5, sample_image * 0.8]

            # Resize all images
            resized_list = px.resize(images, dsize=(50, 50))

            assert isinstance(resized_list, list)
            assert len(resized_list) == 3
            for resized in resized_list:
                assert resized.shape == (50, 50, 3)
