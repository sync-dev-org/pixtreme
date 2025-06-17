import timeit

import cupy as cp
import cv2
import numpy as np

import pixtreme as px


def _test_gaussian():
    print()

    itr = 1000

    with px.Device(0):

        start = timeit.default_timer()
        image_cp = px.imread("example/example.png")
        image_cp = px.to_float32(image_cp)

        image_cp = px.resize(image_cp, (1024, 1024), interpolation=px.INTER_MITCHELL)

        kernel_size = 31
        sigma = 10.0
        blur = px.GaussianBlur()

        start = timeit.default_timer()
        blurred_image_cp = None
        for i in range(itr):
            blurred_image_cp = blur.get(image_cp, kernel_size, sigma)
        assert blurred_image_cp is not None
        assert isinstance(blurred_image_cp, cp.ndarray)
        end = timeit.default_timer()
        total_time = end - start
        fps = itr / total_time
        per_frame = total_time / itr
        print(
            f"Gaussian Blur (Cupy class): Time: {total_time:.6f} seconds, FPS: {fps:.2f}, Per Frame: {per_frame:.6f} seconds"
        )

        # px.imshow("Gaussian Blur (Cupy class)", blurred_image_cp)

        start = timeit.default_timer()
        blurred_image_cp = None
        for i in range(itr):
            blurred_image_cp = px.gaussian_blur(image_cp, kernel_size, sigma)
        assert blurred_image_cp is not None
        assert isinstance(blurred_image_cp, cp.ndarray)
        end = timeit.default_timer()
        total_time = end - start
        fps = itr / total_time
        per_frame = total_time / itr
        print(
            f"Gaussian Blur (Cupy function): Time: {total_time:.6f} seconds, FPS: {fps:.2f}, Per Frame: {per_frame:.6f} seconds"
        )

        # px.imshow("Gaussian Blur (Cupy function)", blurred_image_cp)

        image_np = px.to_numpy(image_cp)

        start = timeit.default_timer()
        blurred_image_np = None
        for i in range(itr):
            blurred_image_np = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), sigma)
        assert blurred_image_np is not None
        assert isinstance(blurred_image_np, np.ndarray)
        end = timeit.default_timer()
        total_time = end - start
        fps = itr / total_time
        per_frame = total_time / itr
        print(
            f"Gaussian Blur (OpenCV): Time: {total_time:.6f} seconds, FPS: {fps:.2f}, Per Frame: {per_frame:.6f} seconds"
        )

        # px.imshow("Gaussian Blur (OpenCV)", blurred_image_np)

        # px.waitkey(0)
        # px.destroy_all_windows()


def test_gaussian():
    _test_gaussian()
