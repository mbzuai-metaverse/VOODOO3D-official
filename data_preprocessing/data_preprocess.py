import torch
import torchvision.transforms as transforms

import cv2
import face_alignment
import numpy as np
from PIL import Image

from data_preprocessing.lm_based_image_align import LandmarkBasedImageAlign
from data_preprocessing.pose_estimation import PoseEstimator
from data_preprocessing.crop_image import ImageCropper
from data_preprocessing.foreground_extractor import ForegroundExtractor


class DataPreprocessor:
    def __init__(self, device, crop_smooth_alpha=0.9):
        self.device = device
        self.crop_smooth_alpha = crop_smooth_alpha

        self.face_alignment = LandmarkBasedImageAlign(output_size=1024, transform_size=1024)
        self.pose_estimator = PoseEstimator(device)
        self.cropper = ImageCropper()
        self.foreground_extractor = ForegroundExtractor(device)
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False, device=device
        )

        self.transform = transforms.ToTensor()

        self.lm_woAlign_ema = None
        self.aligned_lm_ema = None

    def __call__(self, img, keep_bg=False):
        lm = self.fa.get_landmarks(np.array(img))

        if lm is None:
            detected_face = [0, 0, img.size[0], img.size[1]]
            lm = self.fa.get_landmarks(img, detected_faces=[detected_face])[0]
        else:
            lm = lm[0]
        if self.lm_woAlign_ema is None:
            self.lm_woAlign_ema = np.array(lm)
        self.lm_woAlign_ema = (self.lm_woAlign_ema * 0.9 + np.array(lm) * 0.1)
        img_woAlign, lm_woAlign = self.cropper(img, self.lm_woAlign_ema)
        lm_woAlign = np.array(lm_woAlign)
        img_woAlign = np.array(img_woAlign)

        img, aligned_lm = self.face_alignment(img, lm)
        intrinsics, pose = self.pose_estimator.predict_pose(img, aligned_lm)
        img, aligned_lm = self.cropper(img, aligned_lm)
        img = np.array(img)

        if not keep_bg:
            matte = self.foreground_extractor(img)
            img = (img * matte).astype(np.uint8)

        if self.aligned_lm_ema is None:
            self.aligned_lm_ema = aligned_lm
        self.aligned_lm_ema = 0.8 * self.aligned_lm_ema + aligned_lm * 0.2
        crop_params = cv2.estimateAffinePartial2D(self.aligned_lm_ema, lm_woAlign)[0]

        return img, img_woAlign, intrinsics, pose, crop_params

    def from_path(self, image_path, device, keep_bg=False):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        img, _, intrinsics, pose, crop_params = self(img, keep_bg=keep_bg)

        img = np.transpose(img, (2, 0, 1))[None, :, :, :] / 255.
        img = (img * 2 - 1)
        img = torch.from_numpy(img).float()

        pose = torch.from_numpy(pose).unsqueeze(0).float()
        intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).float()

        return {
            'image': img.to(device),
            'cam2world': pose.to(device),
            'intrinsics': intrinsics.to(device),
            'crop_params': crop_params
        }

    @staticmethod
    def realign(img, T):
        out = cv2.warpAffine(img, T, (img.shape[1], img.shape[0]))
        return out
