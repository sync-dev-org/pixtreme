import os
import time
import timeit

import cupy as cp
import cv2
import numpy as np
import pytest

import pixtreme as px

# import pixtreme_source as px


def print_result(method_name, start, end, itr):
    total_time = end - start
    per_time = total_time / itr
    fps = 1 / per_time

    print(f"{method_name:<28}: itr:{itr}, per:{per_time*1000:.2f} ms, {fps:.2f}fps, total:{total_time*1000:.2f} ms")
    time.sleep(1)


@pytest.fixture()
def image_path() -> str:
    path = "example/example.png"

    for i in range(5):
        if os.path.exists(path):
            break
        path = os.path.join("..", path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    return path


@pytest.fixture()
def itr() -> int:
    return 1000


def test_resize(image_path: str, itr: int):
    print()

    with px.Device(0):
        image_cp = px.imread(image_path)
        assert isinstance(image_cp, cp.ndarray)
        image_cp = px.to_float32(image_cp)
        image_np = px.to_numpy(image_cp)
        assert isinstance(image_np, np.ndarray)
        height, width, channels = image_cp.shape

        new_height = height // 2
        new_width = width // 2

        start = timeit.default_timer()
        resized_image_cp_area_c = None
        for _ in range(itr):
            resized_image_cp_area_c = px.resize(image_cp, (new_width, new_height), interpolation=px.INTER_AREA)
        end = timeit.default_timer()
        print_result("px.INTER_AREA", start, end, itr)

        start = timeit.default_timer()
        resized_image_np_area = None
        for _ in range(itr):
            resized_image_np_area = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
        end = timeit.default_timer()
        print_result("cv2.INTER_AREA", start, end, itr)

        new_height = height * 2
        new_width = width * 2

        start = timeit.default_timer()
        resized_image_cp_cubic_c = None
        for _ in range(itr):
            resized_image_cp_cubic_c = px.resize(image_cp, (new_width, new_height), interpolation=px.INTER_CUBIC)
        end = timeit.default_timer()
        print_result("px.INTER_CUBIC", start, end, itr)

        start = timeit.default_timer()
        resized_image_np_cubic = None
        for _ in range(itr):
            resized_image_np_cubic = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        end = timeit.default_timer()
        print_result("cv2.INTER_CUBIC", start, end, itr)

        start = timeit.default_timer()
        resized_image_cp_lanczos4_c = None
        for _ in range(itr):
            resized_image_cp_lanczos4_c = px.resize(image_cp, (new_width, new_height), interpolation=px.INTER_LANCZOS4)
        end = timeit.default_timer()
        print_result("px.INTER_LANCZOS4", start, end, itr)

        start = timeit.default_timer()
        resized_image_np_lanczos4 = None
        for _ in range(itr):
            resized_image_np_lanczos4 = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        end = timeit.default_timer()
        print_result("cv2.INTER_LANCZOS4", start, end, itr)

        start = timeit.default_timer()
        resized_image_cp_nearest_c = None
        for _ in range(itr):
            resized_image_cp_nearest_c = px.resize(image_cp, (new_width, new_height), interpolation=px.INTER_NEAREST)
        end = timeit.default_timer()
        print_result("px.INTER_NEAREST", start, end, itr)

        start = timeit.default_timer()
        resized_image_np_nearest = None
        for _ in range(itr):
            resized_image_np_nearest = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        end = timeit.default_timer()
        print_result("cv2.INTER_NEAREST", start, end, itr)

        start = timeit.default_timer()
        resized_image_cp_linear_c = None
        for _ in range(itr):
            resized_image_cp_linear_c = px.resize(image_cp, (new_width, new_height), interpolation=px.INTER_LINEAR)
        end = timeit.default_timer()
        print_result("px.INTER_LINEAR", start, end, itr)

        start = timeit.default_timer()
        resized_image_np_linear = None
        for _ in range(itr):
            resized_image_np_linear = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        end = timeit.default_timer()
        print_result("cv2.INTER_LINEAR", start, end, itr)

        start = timeit.default_timer()
        resized_image_cp_mitchell_c = None
        for _ in range(itr):
            resized_image_cp_mitchell_c = px.resize(image_cp, (new_width, new_height), interpolation=px.INTER_MITCHELL)
        end = timeit.default_timer()
        print_result("px.INTER_MITCHELL", start, end, itr)

        start = timeit.default_timer()
        resized_image_cp_catmull_rom_c = None
        for _ in range(itr):
            resized_image_cp_catmull_rom_c = px.resize(
                image_cp, (new_width, new_height), interpolation=px.INTER_CATMULL_ROM
            )
        end = timeit.default_timer()
        print_result("px.INTER_CATMULL_ROM", start, end, itr)

        start = timeit.default_timer()
        resized_image_cp_b_spline_c = None
        for _ in range(itr):
            resized_image_cp_b_spline_c = px.resize(image_cp, (new_width, new_height), interpolation=px.INTER_B_SPLINE)
        end = timeit.default_timer()
        print_result("px.INTER_B_SPLINE", start, end, itr)

        start = timeit.default_timer()
        resized_image_cp_lanczos2_c = None
        for _ in range(itr):
            resized_image_cp_lanczos2_c = px.resize(image_cp, (new_width, new_height), interpolation=px.INTER_LANCZOS2)
        end = timeit.default_timer()
        print_result("px.INTER_LANCZOS2", start, end, itr)

        start = timeit.default_timer()
        resized_image_np_lanczos3_c = None
        for _ in range(itr):
            resized_image_np_lanczos3_c = px.resize(image_cp, (new_width, new_height), interpolation=px.INTER_LANCZOS3)
        end = timeit.default_timer()
        print_result("px.INTER_LANCZOS3", start, end, itr)

        # px.imshow("resized_image_cp_area_c", resized_image_cp_area_c)
        # px.imshow("resized_image_np_area", resized_image_np_area)
        # px.imshow("resized_image_cp_cubic_c", resized_image_cp_cubic_c)
        # px.imshow("resized_image_np_cubic", resized_image_np_cubic)
        # px.imshow("resized_image_cp_lanczos4_c", resized_image_cp_lanczos4_c)
        # px.imshow("resized_image_np_lanczos4", resized_image_np_lanczos4)
        # px.imshow("resized_image_cp_nearest_c", resized_image_cp_nearest_c)
        # px.imshow("resized_image_np_nearest", resized_image_np_nearest)
        # px.imshow("resized_image_cp_linear_c", resized_image_cp_linear_c)
        # px.imshow("resized_image_np_linear", resized_image_np_linear)
        # px.imshow("resized_image_cp_mitchell_c", resized_image_cp_mitchell_c)
        # px.imshow("resized_image_cp_catmull_rom_c", resized_image_cp_catmull_rom_c)
        # px.imshow("resized_image_cp_b_spline_c", resized_image_cp_b_spline_c)
        # px.imshow("resized_image_cp_lanczos2_c", resized_image_cp_lanczos2_c)
        # px.imshow("resized_image_np_lanczos3_c", resized_image_np_lanczos3_c)

        # px.waitkey(0)
        # px.destroy_all_windows()
