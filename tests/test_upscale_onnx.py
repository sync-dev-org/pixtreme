import timeit

import cupy as cp

import pixtreme as px


def test_upscale_onnx():
    print()

    device_id = 1
    itr = 8

    with px.Device(device_id):

        image: cp.ndarray = px.imread("example/input/albrecht-voss-x2d-xcd20-35e-1_HD.png")
        image = px.to_float32(image)

        images = [image.copy() for _ in range(itr)]

        onnx_model_path = "models/2xBHI_small_compact_pretrain_1080p_fp16.onnx"
        model = px.OnnxUpscaler(model_path=onnx_model_path, device_id=device_id)

        start = timeit.default_timer()
        up_images = model.get(images)
        end = timeit.default_timer()

        fps = itr / (end - start)
        per_iter_time = (end - start) / itr * 1000
        print("----------------------------------------------------")
        print(f"Interpolation: {model.__class__.__name__}")
        print(f"Time taken: {end - start:.4f} seconds for {itr} iterations")
        print(f"Per iteration time: {per_iter_time:.6f} ms")
        print(f"FPS: {fps:.2f}")

        # px.imshow("up_image", up_images[0])
        # px.waitkey(0)
        # px.destroy_all_windows()
