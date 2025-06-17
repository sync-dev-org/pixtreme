import gc
import os
import timeit

import cupy as cp
import pytest
import torch

import pixtreme as px


@pytest.mark.asyncio
async def test_upscale_trt():
    print()

    device_id = 1
    itr = 300

    with px.Device(device_id):
        input_dir = "example/source/input"
        files = os.listdir(input_dir)

        images = []
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                image: cp.ndarray = px.imread(os.path.join(input_dir, file))
                image = px.to_float32(image)
                image = px.resize(image, dsize=(1280, 720), interpolation=px.INTER_AREA)
                images.append(image)

        height, width, channels = images[0].shape

        pth_model_path = "models/2x-spanx2-ch48.pth"
        # pth_model_path = "models/2xBHI_small_compact_pretrain.pth"

        onnx_model_path = pth_model_path.replace(".pth", f".{height}x{width}.onnx")
        trt_model_path = pth_model_path.replace(".pth", f".{height}x{width}.trt")

        if not os.path.exists(onnx_model_path):
            px.torch_to_onnx(
                pth_model_path,
                onnx_model_path,
                input_shape=(1, channels, height, width),
                opset_version=20,
                precision="fp32",
                dynamic_axes=None,
                device="cuda",
            )

        if not os.path.exists(trt_model_path):
            px.onnx_to_trt(
                onnx_model_path,
                trt_model_path,
                precision="fp16",
                workspace=1 << 20,
            )

        # VRAM Cache Clear bt torch
        torch.cuda.empty_cache()

        # VRAM Cache Clear by cupy

        model = px.TrtUpscaler(model_path=trt_model_path, device_id=device_id)

        start = timeit.default_timer()
        up_images = model.get(images)
        end = timeit.default_timer()

        fps = itr / (end - start)
        per_iter_time = (end - start) / itr * 1000

        for i, image in enumerate(images):
            print(f"Processing image {i + 1}/{len(images)}: {image.shape}")
            print("----------------------------------------------------")
            _image = px.to_uint16(image)
            output_dir = f"example/source/output/{height}x{width}"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"upscaled_{i + 1}.png")
            px.imwrite(output_path=output_path, image=_image)

        up_image = up_images[0]

        del up_images
        gc.collect()
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()

        print("----------------------------------------------------")
        print(f"Time taken: {end - start:.4f} seconds for {itr} iterations")
        print(f"Per iteration time: {per_iter_time:.6f} ms")
        print(f"FPS: {fps:.2f}")

        # px.imshow("up_image", up_image)
        # px.waitkey(0)
        # px.destroy_all_windows()
