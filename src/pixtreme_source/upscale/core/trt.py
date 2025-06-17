from typing import Optional

import cupy as cp
import tensorrt as trt


class TrtUpscaler:
    def __init__(
        self, model_path: Optional[str] = None, model_bytes: Optional[bytes] = None, device_id: int = 0
    ) -> None:
        """
        Initialize the TrtUpscaler.

        Args:
            model_path (str): Path to the TensorRT engine file or ONNX model.
        Raises:
            FileNotFoundError: If the specified path does not exist.
        """

        self.device_id = device_id

        with cp.cuda.Device(self.device_id):
            # Load the TensorRT engine
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)

            if model_path is not None:
                with open(model_path, "rb") as f:
                    model_bytes = f.read()

            if model_bytes is None:
                raise ValueError("model_bytes must be provided if model_path is not specified")

            self.engine = self.runtime.deserialize_cuda_engine(model_bytes)
            self.ctx = self.engine.create_execution_context()

            print("engine loaded ✔  tensors:", self.engine)

            self.initialize()

    def initialize(self):
        """Initialize processing"""
        # Prepare CuPy stream and device buffers
        self.stream = cp.cuda.Stream()
        self.d_inputs = {}
        self.d_outputs = {}

        # Automatically get tensor names
        input_names = []
        output_names = []

        for name in self.engine:
            shape = self.ctx.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            # Convert TensorRT shape to list for CuPy compatibility
            shape_list = [int(dim) for dim in shape]

            # Create a CuPy array for the tensor
            d_arr = cp.empty(shape_list, dtype=dtype)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.d_inputs[name] = d_arr
                input_names.append(name)
            else:
                self.d_outputs[name] = d_arr
                output_names.append(name)

        # Use the first input/output tensor names
        self.input_tensor = input_names[0] if input_names else "input"
        self.output_tensor = output_names[0] if output_names else "output"

        # Set tensor addresses only once during initialization
        for name in self.engine:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.ctx.set_tensor_address(name, self.d_inputs[name].data.ptr)
            else:
                self.ctx.set_tensor_address(name, self.d_outputs[name].data.ptr)

        # Buffer pool
        self.input_buffer = None
        self.output_buffer = None
        self.last_input_shape = None

    def pre_process(self, input_frame: cp.ndarray) -> cp.ndarray:
        """Execute preprocessing in-place"""

        input_shape = input_frame.shape

        channels = 3 if input_shape[-1] >= 3 else input_shape[-1]
        processed_shape = (1, channels, input_shape[0], input_shape[1])

        if self.input_buffer is None or self.input_buffer.shape != processed_shape:
            self.input_buffer = cp.empty(processed_shape, dtype=cp.float32)

        # Check the most common case (RGB) first
        if input_frame.shape[-1] == 3:
            # Process BGR → RGB + HWC → NCHW at once
            self.input_buffer[0, 0] = input_frame[:, :, 2]  # B → R
            self.input_buffer[0, 1] = input_frame[:, :, 1]  # G → G
            self.input_buffer[0, 2] = input_frame[:, :, 0]  # R → B
        elif input_frame.shape[-1] == 4:
            # RGBA → RGB + BGR → RGB + HWC → NCHW
            self.input_buffer[0, 0] = input_frame[:, :, 2]  # B → R
            self.input_buffer[0, 1] = input_frame[:, :, 1]  # G → G
            self.input_buffer[0, 2] = input_frame[:, :, 0]  # R → B
        elif input_frame.ndim == 2:
            # Grayscale → RGB
            self.input_buffer[0, 0] = input_frame
            self.input_buffer[0, 1] = input_frame
            self.input_buffer[0, 2] = input_frame

        return self.input_buffer

    def post_process(self, output_frame: cp.ndarray) -> cp.ndarray:
        """Optimized post-processing"""
        # Don't allocate output buffer each time, only minimal copying
        output_shape = (output_frame.shape[2], output_frame.shape[3], output_frame.shape[1])

        if self.output_buffer is None or self.output_buffer.shape != output_shape:
            # Explicitly delete old buffer
            if self.output_buffer is not None:
                del self.output_buffer
            # Allocate new buffer
            self.output_buffer = cp.empty(output_shape, dtype=output_frame.dtype)

        # NCHW → HWC + RGB → BGR (in-place)
        self.output_buffer[:, :, 0] = output_frame[0, 2]  # R → B
        self.output_buffer[:, :, 1] = output_frame[0, 1]  # G → G
        self.output_buffer[:, :, 2] = output_frame[0, 0]  # B → R

        return self.output_buffer

    def get(self, input_frame: cp.ndarray | list[cp.ndarray]) -> cp.ndarray | list[cp.ndarray]:
        if isinstance(input_frame, list):
            # Use unified buffer for batch processing
            result = []
            for frame in input_frame:
                output_frame = self._get(frame)
                # Immediately copy results and release original buffer
                result.append(cp.copy(output_frame))
        else:
            result = self._get(input_frame)

        # Synchronize (wait for inference completion)
        self.stream.synchronize()

        return result

    def _get(self, input_frame: cp.ndarray) -> cp.ndarray:
        # In-place preprocessing
        processed_input = self.pre_process(input_frame)

        # Copy input data to GPU buffer (minimal shape checking)
        in_gpu = self.d_inputs[self.input_tensor]
        if in_gpu.shape != processed_input.shape:
            # Explicitly delete old buffer
            del self.d_inputs[self.input_tensor]
            # Allocate new buffer
            in_gpu = cp.empty(processed_input.shape, dtype=processed_input.dtype)
            self.d_inputs[self.input_tensor] = in_gpu
            # Reset tensor address
            self.ctx.set_tensor_address(self.input_tensor, in_gpu.data.ptr)

        # Copy data
        in_gpu[:] = processed_input

        # Execute the TensorRT engine asynchronously
        self.ctx.execute_async_v3(self.stream.ptr)

        # Get output data (already on GPU)
        out_gpu = self.d_outputs[self.output_tensor]

        return self.post_process(out_gpu)
