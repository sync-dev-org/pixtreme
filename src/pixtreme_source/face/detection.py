import cupy as cp
import numpy as np
import onnxruntime

from ..color.bgr import bgr_to_rgb, rgb_to_bgr
from ..transform.affine import crop_from_kps
from ..transform.resize import INTER_AUTO, resize
from ..utils.blob import to_blob
from ..utils.dlpack import to_cupy, to_numpy
from ..utils.dtypes import to_float32
from .schema import Face


class FaceDetection:
    def __init__(
        self,
        model_path: str | None = None,
        model_bytes: bytes | None = None,
        device_id: int = 0,
        device: str = "cuda",
    ) -> None:
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

            self.input_size = (640, 640)
            self.center_cache = {}
            self.nms_thresh = 0.4
            self.det_thresh = 0.5
            self._init_vars()
        except Exception as e:
            raise e

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if not isinstance(input_shape[2], str):
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        # self.input_mean = 127.5
        # self.input_std = 128.0

        self.input_mean = 127.5 / 255.0
        self.input_std = 128.0 / 255.0

        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1

        # print(len(outputs))

        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, image, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        mean_val = float(self.input_mean)

        # blob = cv2.dnn.blobFromImage(image, 1.0 / self.input_std, input_size, mean_val, swapRB=True)
        blob = to_blob(image, scalefactor=1.0 / self.input_std, size=self.input_size, mean=(mean_val, mean_val, mean_val))
        blob = to_numpy(blob)

        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        net_outs = [to_cupy(o) for o in net_outs]

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx].astype(cp.float32)
            bbox_preds = net_outs[idx + fmc].astype(cp.float32)
            bbox_preds = bbox_preds * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = cp.stack(cp.mgrid[:height, :width][::-1], axis=-1).astype(cp.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = cp.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = cp.where(scores >= threshold)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2].astype(cp.float32) * stride
                kpss = self.distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def get(self, image, crop_size=512, max_num=0, metric="default") -> list[Face]:
        if isinstance(image, np.ndarray):
            is_np = True
        else:
            is_np = False

        image = to_cupy(image)
        image = bgr_to_rgb(image)
        image = to_float32(image)

        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(self.input_size[1]) / self.input_size[0]
        if im_ratio > model_ratio:
            new_height = self.input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = self.input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = cp.float32(new_height) / image.shape[0]

        resized_image = resize(image, (new_width, new_height))

        det_image = cp.zeros((self.input_size[1], self.input_size[0], 3), dtype=cp.float32)
        det_image[:new_height, :new_width, :] = resized_image

        scores_list, bboxes_list, kpss_list = self.forward(det_image, self.det_thresh)

        scores = cp.vstack(scores_list).astype(cp.float32)

        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        bboxes = cp.vstack(bboxes_list).astype(cp.float32) / det_scale

        pre_det = cp.hstack((bboxes, scores))

        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = cp.vstack(kpss_list).astype(cp.float32) / det_scale
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            image_center = cp.array([image.shape[0] // 2, image.shape[1] // 2], dtype=cp.float32)

            offsets = cp.vstack(
                [(det[:, 0] + det[:, 2]) / 2 - image_center[1], (det[:, 1] + det[:, 3]) / 2 - image_center[0]]
            ).astype(cp.float32)

            offset_dist_squared = cp.sum(cp.power(offsets, 2.0), 0)

            if metric == "max":
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = cp.argsort(values)[::-1]  # some extra weight on the centering

            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        bboxes = det
        results = []
        image = rgb_to_bgr(image)

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :4]
            score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]

            assert kps is not None
            # bbox = bbox.astype(np.float32)
            bbox = bbox.astype(cp.float32)
            cropped_image, M = self.crop(image, kps, size=crop_size)

            if is_np:
                bbox = to_numpy(bbox)
                kps = to_numpy(kps)
                cropped_image = to_numpy(cropped_image)
                M = to_numpy(M)

            face = Face(bbox=bbox, score=score, kps=kps, image=cropped_image, matrix=M)
            results.append(face)

        return results

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = cp.maximum(x1[i], x1[order[1:]]).astype(cp.float32)
            yy1 = cp.maximum(y1[i], y1[order[1:]]).astype(cp.float32)
            xx2 = cp.minimum(x2[i], x2[order[1:]]).astype(cp.float32)
            yy2 = cp.minimum(y2[i], y2[order[1:]]).astype(cp.float32)

            w = cp.maximum(0.0, xx2 - xx1 + 1)
            h = cp.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = cp.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def distance2bbox(self, points: cp.ndarray, distance: cp.ndarray, max_shape=None) -> cp.ndarray:
        points = points.astype(cp.float32)
        distance = distance.astype(cp.float32)

        x1, y1 = points[:, 0] - distance[:, 0], points[:, 1] - distance[:, 1]
        x2, y2 = points[:, 0] + distance[:, 2], points[:, 1] + distance[:, 3]

        if max_shape is not None:
            x1, x2 = cp.clip(x1, 0, max_shape[1]), cp.clip(x2, 0, max_shape[1])
            y1, y2 = cp.clip(y1, 0, max_shape[0]), cp.clip(y2, 0, max_shape[0])

        return cp.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points: cp.ndarray, distance: cp.ndarray, max_shape=None) -> cp.ndarray:
        points = points.astype(cp.float32)
        distance = distance.astype(cp.float32)

        preds = [
            (
                cp.clip(points[:, i % 2] + distance[:, i], 0, max_shape[1] if max_shape else cp.inf)
                if i % 2 == 0
                else cp.clip(points[:, i % 2] + distance[:, i], 0, max_shape[0] if max_shape else cp.inf)
            )
            for i in range(distance.shape[1])
        ]
        return cp.stack(preds, axis=-1)

    def crop(self, image: cp.ndarray, kps: cp.ndarray, size: int = 512) -> tuple:
        output_image, matrix = crop_from_kps(image, kps, size * 2)
        output_image = resize(output_image, (size, size), interpolation=INTER_AUTO)
        matrix /= 2
        return output_image, matrix
