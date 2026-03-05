import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List

class InterpretationVisualizer:
    """
    Tools for visualizing attribution maps and heatmaps.
    """
    
    @staticmethod
    def generate_heatmap(all_regions: List[np.ndarray], 
                         selected_indices: List[int], 
                         scores: List[float], 
                         image_shape: tuple) -> np.ndarray:
        """
        Generates a cumulative saliency map from selected regions and their scores.
        image_shape: (H, W, ...) or (T, H, W, ...)
        """
        # If it's video (T, H, W, 1), image_shape will have 4 dims
        is_video = len(image_shape) >= 4 or (len(all_regions) > 0 and all_regions[0].ndim == 4)
        
        if is_video:
            t, h, w = all_regions[0].shape[:3]
            heatmap = np.zeros((t, h, w), dtype=np.float32)
        else:
            h, w = image_shape[:2]
            heatmap = np.zeros((h, w), dtype=np.float32)
        
        prev_score = 0
        for idx, score in zip(selected_indices, scores):
            mask = all_regions[idx].squeeze()
            gain = max(0, score - prev_score)
            heatmap[mask == 1] += gain
            prev_score = score
            
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
        return heatmap

    @staticmethod
    def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha=0.5) -> np.ndarray:
        """
        Overlays a heatmap on an image or video.
        image: (H, W, 3) or (T, H, W, 3)
        heatmap: (H, W) or (T, H, W)
        """
        if image.ndim == 4: # Video
            t, h, w, _ = image.shape
            overlays = []
            for i in range(t):
                frame_overlay = InterpretationVisualizer.overlay_heatmap(image[i], heatmap[i], alpha)
                overlays.append(frame_overlay)
            return np.stack(overlays)
        
        # Single image logic
        h, w, _ = image.shape
...

    @staticmethod
    def save_comparison(image: np.ndarray, overlay: np.ndarray, heatmap: np.ndarray, save_path: str):
        """
        Saves a side-by-side comparison for image or video.
        If video, it saves a grid-based video file.
        """
        if image.ndim == 4: # Video
            return InterpretationVisualizer.save_video_comparison(image, overlay, heatmap, save_path)
        
        # Original single image logic ...
        plt.figure(figsize=(15, 5))
...
        plt.close()
        print(f"[Visualizer] Saved visualization to {save_path}")

    @staticmethod
    def save_video_comparison(video: np.ndarray, overlay: np.ndarray, heatmap: np.ndarray, save_path: str, fps=2):
        """
        Saves a 3-panel comparison video (Original | Heatmap | Overlay).
        """
        t, h, w, _ = video.shape
        # Prepare heatmap for visualization (apply colormap)
        heatmap_color = []
        for i in range(t):
            h_frame = cv2.applyColorMap(np.uint8(255 * heatmap[i]), cv2.COLORMAP_JET)
            heatmap_color.append(h_frame)
        heatmap_color = np.stack(heatmap_color)
        
        # Combine side-by-side: [T, H, W*3, 3]
        combined = np.concatenate([video, heatmap_color, overlay], axis=2)
        
        # Save using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (w * 3, h))
        
        for i in range(t):
            out.write(combined[i].astype(np.uint8))
        out.release()
        print(f"[Visualizer] Saved video visualization to {save_path}")
