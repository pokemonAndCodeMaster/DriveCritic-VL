import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from ..processors.visual_processor import VisualProcessor

class EagleStrategy:
    """
    Implements the EAGLE (Efficient Attribution for Grounded Language Explanations) strategy.
    Uses submodular optimization to find the most influential visual regions.
    """
    def __init__(self, adaptor, lambda1=1.0, lambda2=1.0, batch_size=8):
        self.adaptor = adaptor
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.batch_size = batch_size # How many candidates to evaluate in one forward pass
        self.device = adaptor.model_device if hasattr(adaptor, "model_device") else "cuda"

    def run(self, source_image: np.ndarray, all_regions: List[np.ndarray], max_steps: int = None) -> Tuple[List[int], List[float]]:
        """
        Executes the greedy search to select regions that maximize the submodular objective.
        
        Args:
            source_image: Original image (H, W, 3) BGR.
            all_regions: List of masks (H, W, 1).
            max_steps: Maximum number of regions to select. Defaults to len(all_regions).
            
        Returns:
            selected_indices: Ordered list of selected region indices.
            attribution_scores: The submodular score at each selection step.
        """
        n = len(all_regions)
        if max_steps is None:
            max_steps = n
        else:
            max_steps = min(max_steps, n)
            
        selected_indices = []
        remaining_indices = list(range(n))
        attribution_scores = []
        
        # Pre-calculate tensors for fast masking
        # BGR source image tensor
        source_tensor = torch.from_numpy(source_image).float().to(self.device) # [H, W, 3]
        # All masks as a 4D tensor [N, H, W, 1]
        masks_tensor = torch.from_numpy(np.array(all_regions)).float().to(self.device)
        
        # Current selected mask S
        s_mask = torch.zeros((source_image.shape[0], source_image.shape[1], 1), device=self.device)
        
        print(f"[EAGLE] Starting greedy search for {max_steps} steps (Total regions: {n})...")

        for step in range(max_steps):
            candidate_scores = []
            
            num_candidates = len(remaining_indices)
            
            # Use tqdm for outer loop if batch size is small
            for i in tqdm(range(0, num_candidates, self.batch_size), desc=f"Step {step+1}", leave=False):
                batch_rem_indices = remaining_indices[i : i + self.batch_size]
                
                # Each candidate c generates two images:
                # 1. Insertion: S U {c}
                # 2. Deletion: V \ (S U {c})
                
                # Get masks for candidates in this batch
                c_masks = masks_tensor[batch_rem_indices] # [B, H, W, 1]
                
                # Calculate Insertion masks: S + c
                ins_masks = (s_mask.unsqueeze(0) + c_masks).clamp(0, 1) # [B, H, W, 1]
                
                # Calculate Deletion masks: 1 - (S + c)
                del_masks = (1.0 - ins_masks) # [B, H, W, 1]
                
                # Combine to create a single batch of images for the adaptor
                # We alternate [Ins_0, Del_0, Ins_1, Del_1, ...]
                combined_masks = torch.empty((len(batch_rem_indices) * 2, *ins_masks.shape[1:]), device=self.device)
                combined_masks[0::2] = ins_masks
                combined_masks[1::2] = del_masks
                
                # Generate masked images using broadcasting
                # (batch*2, H, W, 1) * (1, H, W, 3) = (batch*2, H, W, 3)
                batch_images_tensor = combined_masks * source_tensor.unsqueeze(0)
                
                # We pass the tensor batch to the adaptor
                # NOTE: Adaptor must handle this tensor batch!
                all_probs = self.adaptor.forward(list(batch_images_tensor))
                
                # Separate insertion and deletion probabilities
                ins_probs = all_probs[0::2]
                del_probs = all_probs[1::2]
                
                # Calculate submodular score: f(S U {c}) = lambda1 * ins + lambda2 * (1 - del)
                batch_scores = self.lambda1 * ins_probs + self.lambda2 * (1.0 - del_probs)
                candidate_scores.extend(batch_scores.cpu().numpy().tolist())

            # Find the best candidate in this step
            best_idx_in_remaining = np.argmax(candidate_scores)
            best_global_idx = remaining_indices[best_idx_in_remaining]
            
            # Record selection
            selected_indices.append(best_global_idx)
            remaining_indices.pop(best_idx_in_remaining)
            attribution_scores.append(candidate_scores[best_idx_in_remaining])
            
            # Update current selected mask S
            s_mask = (s_mask + masks_tensor[best_global_idx]).clamp(0, 1)
            
            # Logging progress
            if (step + 1) % 5 == 0 or step == 0:
                print(f"[EAGLE] Step {step+1}/{max_steps}: Selected region {best_global_idx}, Score: {attribution_scores[-1]:.4f}")

        return selected_indices, attribution_scores
