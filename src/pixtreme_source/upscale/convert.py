import os
import sys

import spandrel_extra_arches as ex_arch
import tensorrt as trt
import torch
from spandrel import ModelLoader


def torch_to_onnx(
    model_path: str,
    onnx_path: str,
    input_shape: tuple = (1, 3, 1080, 1920),
    opset_version: int = 20,
    precision: str = "fp32",
    dynamic_axes: dict | None = None,
    device: str = "cuda",
) -> None:

    print(f"Exporting PyTorch model to ONNX: {model_path} → {onnx_path}")
    ex_arch.install()

    model = ModelLoader().load_from_file(model_path).model.to(torch.device(device)).eval()

    if precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    with torch.autocast(device, dtype=dtype):
        dummy_input = torch.randn(input_shape, device=device)
        torch.onnx.export(
            model,
            (dummy_input,),
            onnx_path,
            opset_version=opset_version,
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            optimize=True,
        )

    print(f"✅ ONNX model exported to: {onnx_path}")


def onnx_to_trt(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    workspace: int = 1024 << 20,
) -> None:
    print(f"Building TensorRT engine from ONNX model: {onnx_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # --- Analyze ONNX model ---
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            raise RuntimeError("Failed to parse the ONNX model")

    # --- BuilderConfig ---
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)

    # --- Build engine ---
    engine_bytes = builder.build_serialized_network(network, config)  # TRT ≥ 8.2 :contentReference[oaicite:3]{index=3}
    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    if not os.path.exists(engine_path):
        raise RuntimeError(f"Failed to save TensorRT engine to {engine_path}")

    print(f"✅ TensorRT engine saved to: {engine_path}")
