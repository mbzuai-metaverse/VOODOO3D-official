import numpy as np
import PIL
from PIL import Image
import cv2
import scipy


class LandmarkBasedImageAlign:
    def __init__(self, output_size, transform_size):
        self.output_size = output_size
        self.transform_size = transform_size

    @staticmethod
    def calc_quad(lm):
        # Calculate auxiliary vectors.
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise

        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        q_scale = 1.8
        x = q_scale * x
        y = q_scale * y
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])

        qsize = np.hypot(*x) * 2

        return quad, qsize

    def _shrink(self, img, lm, quad, qsize):
        shrink = int(np.floor(qsize / self.output_size * 0.5))

        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.LANCZOS)
            lm /= rsize
            quad /= shrink
            qsize /= shrink

        return img, lm, quad, qsize

    def _crop(self, img, lm, quad, qsize):
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1])))
        )
        crop = (
            max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1])
        )
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            lm -= crop[0: 2]
            quad -= crop[0:2]

        return img, lm, quad, qsize

    def _pad(self, img, lm, quad, qsize):
        border = max(int(np.rint(qsize * 0.1)), 3)
        pad = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1])))
        )
        pad = (
            max(-pad[0] + border, 0),
            max(-pad[1] + border, 0),
            max(pad[2] - img.size[0] + border, 0),
            max(pad[3] - img.size[1] + border, 0)
        )

        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.array(img).astype(np.float32), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(
                1.0 - np.minimum(
                    np.float32(x) / pad[0],
                    np.float32(w-1-x) / pad[2]
                ),
                1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3])
            )
            low_res = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
            blur = qsize * 0.02*0.1
            low_res = scipy.ndimage.gaussian_filter(low_res, [blur, blur, 0])
            low_res = cv2.resize(low_res, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

            img += (low_res - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            median = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
            median = np.median(median, axis=(0, 1))
            img += (median - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]
            lm += pad[0: 2]

        return img, lm, quad, qsize

    def _extract_quad(self, img, lm, quad, qsize):
        img_size = np.array([img.size[1], img.size[0]])[None, :]
        quad_center = quad.mean(axis=0)

        lm = lm - img_size / 2
        quad_center = quad_center - img_size / 2
        lm = lm - quad_center

        rotate_angle = 2 * np.pi - np.arctan2(*np.flipud(quad[3] - quad[0]))
        R = np.array([[np.cos(rotate_angle), -np.sin(rotate_angle)], [np.sin(rotate_angle), np.cos(rotate_angle)]])

        lm = lm @ R.T
        lm = lm / qsize * self.transform_size
        lm = lm + np.array([self.transform_size / 2, self.transform_size / 2])[None, :]

        img = img.transform(
            (self.transform_size, self.transform_size),
            PIL.Image.QUAD,
            (quad + 0.5).flatten(),
            PIL.Image.NEAREST
        )

        if self.output_size < self.transform_size:
            lm = lm / self.transform_size * self.output_size
            img = img.resize((self.output_size, self.output_size), PIL.Image.NEAREST)

        return img, lm, quad, qsize

    @staticmethod
    def _debug(img, lm):
        img = np.array(img)

        for v in lm:
            x, y = int(v[0]), int(v[1])
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('debug.png', img)

        assert 0

    def __call__(self, img, lm):
        lm = lm.copy()
        quad, qsize = self.calc_quad(lm)

        img, lm, quad, qsize = self._shrink(img, lm, quad, qsize)
        img, lm, quad, qsize = self._extract_quad(img, lm, quad, qsize)

        return img, lm


if __name__ == '__main__':
    from data_processing.face_detector import FaceDetector
    face_detector = FaceDetector('cuda')
    align_landmarks = LandmarkBasedImageAlign(output_size=1500, num_threads=1, transform_size=1024)

    img = Image.open('test_image.png')

    pred_lm, _ = face_detector(img)
    img_aligned, lm_aligned = align_landmarks.align_image(img, pred_lm)
