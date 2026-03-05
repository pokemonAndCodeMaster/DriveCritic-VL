import cv2
import numpy as np
import torch
from typing import List, Tuple

class VisualProcessor:
    """
    Handles image segmentation and mask generation for attribution.
    Supports both 2D images and 3D video volumes.
    """
    def __init__(self, mode="grid", patch_size=16, region_size=30):
        self.mode = mode
        self.patch_size = patch_size # For grid mode
        self.region_size = region_size # For SLIC mode

    def divide_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Divides the image into several regions (super-pixels or grid patches).
        image: np.ndarray (H, W, 3) BGR or (T, H, W, 3) BGR
        Returns: List of masks (H, W, 1) or (T, H, W, 1).
        """
        if image.ndim == 4: # Video (T, H, W, 3)
            return self.divide_video_regions(image)
            
        h, w, _ = image.shape
        element_sets_V = []

        if self.mode == "slic":
            if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'createSuperpixelSLIC'):
                slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=self.region_size, ruler=20.0)
                slic.iterate(20)
                label_slic = slic.getLabels()
                number_slic = slic.getNumberOfSuperpixels()
                for i in range(number_slic):
                    mask = (label_slic == i)[:, :, np.newaxis].astype(np.uint8)
                    element_sets_V.append(mask)
            else:
                print("[VisualProcessor] SLIC not available, falling back to grid.")
                self.mode = "grid"

        if self.mode == "grid":
            for y in range(0, h, self.patch_size):
                for x in range(0, w, self.patch_size):
                    mask = np.zeros((h, w, 1), dtype=np.uint8)
                    y_end = min(y + self.patch_size, h)
                    x_end = min(x + self.patch_size, w)
                    mask[y:y_end, x:x_end, :] = 1
                    element_sets_V.append(mask)

        return element_sets_V

    def divide_video_regions(self, video_frames: np.ndarray, t_patch_size: int = 1) -> List[np.ndarray]:
        """
        Divides a video into 3D spatio-temporal blocks.
        video_frames: np.ndarray (T, H, W, 3)
        t_patch_size: Number of frames per temporal segment.
        Returns: List of 3D masks (T, H, W, 1)
        """
        t, h, w, _ = video_frames.shape
        element_sets_V = []
        
        # Grid-based division in space and time
        for ti in range(0, t, t_patch_size):
            for yi in range(0, h, self.patch_size):
                for xi in range(0, w, self.patch_size):
                    mask = np.zeros((t, h, w, 1), dtype=np.uint8)
                    t_end = min(ti + t_patch_size, t)
                    y_end = min(yi + self.patch_size, h)
                    x_end = min(xi + self.patch_size, w)
                    mask[ti:t_end, yi:y_end, xi:x_end, :] = 1
                    element_sets_V.append(mask)
        return element_sets_V

    @staticmethod
    def apply_masks(image: np.ndarray, masks: List[np.ndarray], baseline=0) -> np.ndarray:
        """
        Applies a set of masks to an image or video.
        masks: list of (H, W, 1) or (T, H, W, 1) masks.
        """
        if not masks:
            return np.full_like(image, baseline, dtype=np.uint8)
            
        combined_mask = np.zeros_like(masks[0], dtype=np.float32)
        for m in masks:
            combined_mask += m.astype(np.float32)
        combined_mask = np.clip(combined_mask, 0, 1)

        # Broadcast mask to 3 channels
        combined_mask_3c = np.repeat(combined_mask, 3, axis=-1)
        
        masked_image = (image.astype(np.float32) * combined_mask_3c + 
                        baseline * (1.0 - combined_mask_3c)).astype(np.uint8)
        
        return masked_image

    @staticmethod
    def generate_masked_image(source_image: np.ndarray, 
                              all_regions: List[np.ndarray], 
                              selected_indices: List[int], 
                              baseline=0) -> np.ndarray:
        """
        Helper to generate a masked image from a set of region masks and indices.
        """
        selected_masks = [all_regions[i] for i in selected_indices]
        return VisualProcessor.apply_masks(source_image, selected_masks, baseline)
