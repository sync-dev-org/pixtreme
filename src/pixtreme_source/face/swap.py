import numpy as np
import onnxruntime

from ..color.bgr import bgr_to_rgb, rgb_to_bgr
from ..transform.resize import INTER_AUTO, resize
from ..utils.blob import to_blob
from ..utils.dlpack import to_cupy, to_numpy
from .emap import load_emap
from .schema import Face


class FaceSwap:
    def __init__(
        self,
        model_path: str | None = None,
        model_bytes: bytes | None = None,
        device_id: int = 0,
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
            # modelproto = onnx.load_model_from_string(model_bytes)
            # graph = modelproto.graph
            # self.emap = numpy_helper.to_array(graph.initializer[-1])

            self.emap = load_emap()

            self.input_mean = 0.0
            self.input_std = 1.0

            inputs = self.session.get_inputs()
            self.input_names = []
            for inp in inputs:
                self.input_names.append(inp.name)
            outputs = self.session.get_outputs()
            output_names = []
            for out in outputs:
                output_names.append(out.name)
            self.output_names = output_names
            assert len(self.output_names) == 1
            input_cfg = inputs[0]
            input_shape = input_cfg.shape
            self.input_shape = input_shape
            self.input_size = tuple(input_shape[2:4][::-1])
        except Exception as e:
            raise e

    def forward(self, img, latent) -> np.ndarray:
        try:
            img = (img - self.input_mean) / self.input_std
            pred = self.session.run(self.output_names, {self.input_names[0]: img, self.input_names[1]: latent})[0]
            assert isinstance(pred, np.ndarray)
            return pred
        except Exception as e:
            raise e

    def get(self, target_face: Face, source_face: Face) -> np.ndarray:
        try:
            assert target_face.image is not None
            image = target_face.image
            normed_embedding = source_face.normed_embedding
            target_size = image.shape[:2]

            if isinstance(image, np.ndarray):
                is_np = True
            else:
                is_np = False
            image = to_numpy(image)
            normed_embedding = to_numpy(normed_embedding)
            image = bgr_to_rgb(image)

            aimage = resize(image, (128, 128), interpolation=INTER_AUTO)

            blob = to_blob(aimage, 1.0 / self.input_std, (128, 128), (self.input_mean, self.input_mean, self.input_mean))
            latent = normed_embedding.reshape((1, -1))
            latent = np.dot(latent, self.emap)
            latent /= np.linalg.norm(latent)

            pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]

            assert isinstance(pred, np.ndarray)
            img_fake = pred.transpose((0, 2, 3, 1))[0]
            img_fake = np.clip(img_fake, 0, 1).astype(np.float32)
            img_fake = rgb_to_bgr(img_fake)
            img_fake = resize(img_fake, (target_size[1], target_size[0]), interpolation=INTER_AUTO)

            if is_np:
                img_fake = to_numpy(img_fake)
            else:
                img_fake = to_cupy(img_fake)

            return img_fake
        except Exception as e:
            raise e
