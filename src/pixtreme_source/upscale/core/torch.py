from io import BytesIO

import cupy as cp
import torch
from spandrel import Architecture, ImageModelDescriptor, ModelLoader, ModelTiling, SizeRequirements

from ...color.bgr import bgr_to_rgb, rgb_to_bgr
from ...transform.resize import INTER_AUTO, resize
from ...utils.dlpack import to_cupy, to_tensor
from ...utils.dtypes import to_float32


class TorchUpscaler:
    def __init__(
        self,
        model_path: str | None = None,
        model_bytes: bytes | None = None,
        device: str = "cuda",
    ) -> None:
        try:
            if model_bytes is not None:
                _model_bytes = BytesIO(model_bytes)
                state_dict = torch.load(_model_bytes, weights_only=False)
            elif model_path is not None:
                state_dict = torch.load(model_path, weights_only=False)
            else:
                raise ValueError("Model or model_path must be provided")

            self.model = ModelLoader().load_from_state_dict(state_dict)
            assert isinstance(self.model, ImageModelDescriptor)

            if "cuda:" in device:
                self.device_id = int(device.split(":")[-1])
            else:
                self.device_id = 0

            self.device: torch.device = torch.device(device)
            self.model.to(self.device).eval()
            self.scale: int = self.model.scale
            self.size_requirements: SizeRequirements = self.model.size_requirements
            self.tiling: ModelTiling = self.model.tiling
            self.architecture: Architecture = self.model.architecture
            self.purpose: str = self.model.purpose

        except Exception as e:
            raise e

    def pre_process(
        self,
        input_image: cp.ndarray,
    ) -> torch.Tensor:
        with cp.cuda.Device(self.device_id):
            if self.purpose == "FaceSR":
                required_size = self.size_requirements.minimum
                if input_image.shape[:2] != (required_size, required_size):
                    input_image = resize(input_image, (required_size, required_size), interpolation=INTER_AUTO)

            input_image = bgr_to_rgb(input_image)
            input_image = to_float32(input_image)

            input_tensor = to_tensor(input_image, device=self.device)

            return input_tensor

    def post_process(
        self,
        output_tensor: torch.Tensor,
    ) -> cp.ndarray:
        with cp.cuda.Device(self.device_id):
            output_image = to_cupy(output_tensor)
            output_image = rgb_to_bgr(output_image)

            return output_image

    def get(
        self,
        image: cp.ndarray | list[cp.ndarray],
    ) -> cp.ndarray | list[cp.ndarray]:
        if isinstance(image, list):
            return [self._get(img) for img in image]
        return self._get(image)

    def _get(
        self,
        image: cp.ndarray,
    ) -> cp.ndarray:
        with cp.cuda.Device(self.device_id):
            input_tensor: torch.Tensor = self.pre_process(image)

            with torch.no_grad():
                output_tensor: torch.Tensor = self.model.model(input_tensor)

            output_image: cp.ndarray = self.post_process(output_tensor)
            return output_image
