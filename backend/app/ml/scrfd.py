"""
app/ml/scrfd.py

Minimal ONNX Runtime implementation of SCRFD (Sample and Computation
Redistribution for Face Detection), the detector family InsightFace ships
its det_500m / det_10g models as.

SCRFD is an anchor-based, FPN-style detector predicting, at each of three
strides (8, 16, 32), a per-location objectness score, a bounding-box
regression (as distances to the anchor point) and a 5-point facial
landmark regression (also as distances to the anchor point). This module
re-implements the standard decode/NMS steps so the model can be run with
nothing but onnxruntime + numpy + opencv.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

STRIDES = (8, 16, 32)
NUM_ANCHORS = 2  # SCRFD predicts 2 anchors per spatial location, per stride


@dataclass
class Detection:
    bbox: np.ndarray        # [x1, y1, x2, y2] in original image coordinates
    score: float
    landmarks: np.ndarray   # (5, 2) [x, y] in original image coordinates


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    """Standard greedy NMS. dets: (N, 5) = [x1, y1, x2, y2, score]."""
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:][iou <= thresh]

    return keep


class SCRFD:
    """Loads a SCRFD ONNX model and runs detection on BGR numpy images."""

    def __init__(self, session: ort.InferenceSession, input_size: int = 640) -> None:
        self._session = session
        self._input_size = input_size

        self._input_name = session.get_inputs()[0].name
        self._output_names = [o.name for o in session.get_outputs()]
        # Outputs are grouped by shape at decode time rather than assumed
        # export order, since ONNX export tools don't guarantee ordering:
        #   score -> (N, 1)   bbox -> (N, 4)   kps -> (N, 10)

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Letterbox-resize to a square input_size x input_size canvas."""
        h, w = img.shape[:2]
        scale = self._input_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))

        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized

        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 128.0,
            size=(self._input_size, self._input_size),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        return blob, scale

    def detect(
        self,
        img: np.ndarray,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> list[Detection]:
        blob, scale = self._preprocess(img)
        outputs = self._session.run(self._output_names, {self._input_name: blob})

        # Sort outputs into per-stride buckets by matching each output's
        # element count to the expected anchor count for that stride at
        # this input size, and its last dim (1=score, 4=bbox, 10=kps).
        scores_by_stride: dict[int, np.ndarray] = {}
        bboxes_by_stride: dict[int, np.ndarray] = {}
        kps_by_stride: dict[int, np.ndarray] = {}

        for out in outputs:
            arr = np.asarray(out)
            arr = arr.reshape(-1, arr.shape[-1])
            last_dim = arr.shape[-1]
            n = arr.shape[0]

            matched_stride = None
            for stride in STRIDES:
                fmap = self._input_size // stride
                expected_n = fmap * fmap * NUM_ANCHORS
                if n == expected_n:
                    matched_stride = stride
                    break
            if matched_stride is None:
                continue

            if last_dim == 1:
                scores_by_stride[matched_stride] = arr.reshape(-1)
            elif last_dim == 4:
                bboxes_by_stride[matched_stride] = arr
            elif last_dim == 10:
                kps_by_stride[matched_stride] = arr

        all_bboxes = []
        all_scores = []
        all_kps = []

        for stride in STRIDES:
            if stride not in scores_by_stride or stride not in bboxes_by_stride:
                continue

            fmap = self._input_size // stride
            centers = np.stack(np.mgrid[:fmap, :fmap][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            if NUM_ANCHORS > 1:
                centers = np.repeat(centers, NUM_ANCHORS, axis=0)

            scores = scores_by_stride[stride]
            bbox_preds = bboxes_by_stride[stride] * stride
            bboxes = _distance2bbox(centers, bbox_preds)

            mask = scores >= conf_threshold
            if not np.any(mask):
                continue

            all_bboxes.append(bboxes[mask])
            all_scores.append(scores[mask])

            if stride in kps_by_stride:
                kps_preds = kps_by_stride[stride] * stride
                kps = _distance2kps(centers, kps_preds)
                all_kps.append(kps[mask])
            else:
                all_kps.append(np.zeros((mask.sum(), 10), dtype=np.float32))

        if not all_bboxes:
            return []

        bboxes = np.concatenate(all_bboxes, axis=0) / scale
        scores = np.concatenate(all_scores, axis=0)
        kps = np.concatenate(all_kps, axis=0) / scale

        dets = np.concatenate([bboxes, scores[:, None]], axis=1)
        keep = _nms(dets, nms_threshold)

        results = []
        for i in keep:
            results.append(
                Detection(
                    bbox=bboxes[i],
                    score=float(scores[i]),
                    landmarks=kps[i].reshape(5, 2),
                )
            )
        return results
