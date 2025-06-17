import numpy as np
import onnx
import onnxruntime

from ..color.bgr import bgr_to_rgb
from ..utils.blob import to_blobs
from ..utils.dlpack import to_cupy, to_numpy
from ..utils.dtypes import to_float32
from .schema import Face


class FaceEmbedding:
    def __init__(
        self,
        model_path: str | None = None,
        model_bytes: bytes | None = None,
        device: str = "cuda",
    ):
        try:
            sees_options = onnxruntime.SessionOptions()
            sees_options.log_severity_level = 4
            sees_options.log_verbosity_level = 4

            provider_options = [{}]
            if "cuda:" in device:
                providers = ["CUDAExecutionProvider"]
                provider_options = [{"device_id": device.split(":")[-1]}]
            elif device == "cuda":
                providers = ["CUDAExecutionProvider"]
            elif device == "trt":
                providers = ["TensorrtExecutionProvider"]
                provider_options = [
                    {
                        "trt_max_workspace_size": 1073741824 * 2,
                        "trt_fp16_enable": False,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "models/trt_engine_cache",
                        "trt_timing_cache_enable": True,
                        "trt_sparsity_enable": True,
                    }
                ]
            else:
                providers = ["CPUExecutionProvider"]

            onnx_params = {
                "session_options": sees_options,
                "providers": providers,
                "provider_options": provider_options,
            }
            if model_bytes is not None:
                pass
            elif model_path is not None:
                with open(model_path, "rb") as f:
                    model_bytes = f.read()
            else:
                raise ValueError("model or model_path is required")

            self.session = onnxruntime.InferenceSession(model_bytes, **onnx_params)
            modelproto = onnx.load_model_from_string(model_bytes)

            find_sub = False
            find_mul = False

            graph = modelproto.graph
            for nid, node in enumerate(graph.node[:8]):
                # print(nid, node.name)
                if node.name.startswith("Sub") or node.name.startswith("_minus"):
                    find_sub = True
                if node.name.startswith("Mul") or node.name.startswith("_mul"):
                    find_mul = True
            if find_sub and find_mul:
                # mxnet arcface model
                input_mean = 0.0
                input_std = 1.0
            else:
                input_mean = 127.5 / 255.0
                input_std = 127.5 / 255.0
            self.input_mean = input_mean
            self.input_std = input_std
            input_cfg = self.session.get_inputs()[0]
            input_shape = input_cfg.shape
            input_name = input_cfg.name
            self.input_size = tuple(input_shape[2:4][::-1])
            self.input_shape = input_shape
            outputs = self.session.get_outputs()
            output_names = []
            for out in outputs:
                output_names.append(out.name)
            self.input_name = input_name
            self.output_names = output_names
            assert len(self.output_names) == 1
            self.output_shape = outputs[0].shape
        except Exception as e:
            raise e

    def forward(self, batch_data):
        assert self.session is not None
        blob = (batch_data - self.input_mean) / self.input_std
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def get(self, face: Face):
        if isinstance(face.image, np.ndarray):
            is_np = True
        else:
            is_np = False
        _image = to_numpy(face.image)
        _image = bgr_to_rgb(_image)
        _image = to_float32(_image)

        face.embedding = self.get_feat(_image).flatten()
        assert face.embedding is not None

        face.normed_embedding = face.embedding / np.linalg.norm(face.embedding)
        assert face.normed_embedding is not None

        if is_np:
            face.embedding = to_numpy(face.embedding)
            face.normed_embedding = to_numpy(face.normed_embedding)
        else:
            face.embedding = to_cupy(face.embedding)
            face.normed_embedding = to_cupy(face.normed_embedding)

        return face

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size

        # blob = cv2.dnn.blobFromImages(
        #    imgs, 1.0 / self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True
        # )
        blob = to_blobs(imgs, 1.0 / self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean))
        blob = to_numpy(blob)

        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out
