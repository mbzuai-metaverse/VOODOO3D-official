import time
import face_alignment
import numpy as np
import cv2

from DDFA_V2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX


class FaceBoxEstimator:
    def __init__(self, device):
        device_idx = int(device.split(':')[-1])
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': device_idx,
            }),
            'CPUExecutionProvider',
        ]
        self.face_boxes = FaceBoxes_ONNX(providers=providers)
        self.bboxes = []

    def add_box(self, box):
        self.bboxes.append(box)

        if len(self.bboxes) > 5:
            self.bboxes.pop(0)

    def clear_history(self):
        self.bboxes = []

    def get_current_box(self):
        return np.mean(self.bboxes, axis=0)

    def __call__(self, img):
        boxes = self.face_boxes(img)

        if len(boxes) == 0:
            return boxes

        self.add_box(boxes[0])
        return [self.get_current_box()]


class FaceDetector:
    def __init__(self, device):
        self.facebox_detector = FaceBoxEstimator(device)
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False, device=device
        )

        self.keypoints = []

    def add(self, kps):
        self.keypoints.append(kps)

        if len(self.keypoints) > 5:
            self.keypoints.pop(0)

    def clear_history(self):
        self.bboxes = []

    def get_current_keypoints(self):
        return np.mean(self.keypoints, axis=0)

    def __call__(self, img):
        img = np.array(img)
        scale_factor = 512 / max(img.shape[0], img.shape[1])
        img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)

        bbox = self.facebox_detector(img)
        landmarks = self.fa.get_landmarks(img, detected_faces=bbox)

        if landmarks is None:
            return None, None

        landmarks = np.array(landmarks)[0] / scale_factor
        bbox = np.array(bbox)[0] / scale_factor

        self.add(landmarks)
        landmarks = self.get_current_keypoints()

        return landmarks, bbox


if __name__ == '__main__':
    from PIL import Image
    detector = FaceDetector()

    img = Image.open('00000/00008.png')
    lm, bbox = detector(img)
    bbox = bbox.astype(int)

    print('Score:', bbox[-1])
    img = np.array(img)
    cv2.rectangle(img, bbox[:2], bbox[2:4], (255, 0, 0), thickness=5)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    cv2.imshow('debug', img)
    cv2.waitKey(-1)
