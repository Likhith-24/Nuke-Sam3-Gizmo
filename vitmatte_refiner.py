# vitmatte_refiner.py - High-quality alpha matte refinement using ViTMatte
# Converts SAM3 coarse masks into production-quality alpha mattes

import numpy as np
from typing import Tuple, Optional
import cv2


class ViTMatteRefiner:
    """ViTMatte-based alpha matte refinement for high-quality edges."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self._loaded = False
    
    def load(self):
        """Load the ViTMatte model."""
        if self._loaded:
            return
        
        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f"[ViTMatte] Loading model on {self.device}...")
        
        model_name = "hustvl/vitmatte-small-composition-1k"
        
        self.processor = VitMatteImageProcessor.from_pretrained(model_name)
        self.model = VitMatteForImageMatting.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self._loaded = True
        print("[ViTMatte] Model loaded successfully.")
    
    def create_trimap(
        self,
        mask: np.ndarray,
        unknown_width: int = 10,
        erode_radius: int = 2,
        dilate_radius: int = 2
    ) -> np.ndarray:
        """
        Create a trimap from a binary/soft mask.
        
        Args:
            mask: Input mask (0-1 float or 0-255 uint8)
            unknown_width: Width of the uncertain region
            erode_radius: Radius for morphological erosion cleanup
            dilate_radius: Radius for morphological dilation cleanup
        
        Returns:
            Trimap with 0 (BG), 128 (unknown), 255 (FG)
        """
        # Normalize to uint8
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.astype(np.uint8)
        
        # Threshold to binary
        _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        
        # Apply erode/dilate cleanup if specified
        if erode_radius > 0:
            erode_kernel = np.ones((erode_radius * 2 + 1, erode_radius * 2 + 1), np.uint8)
            binary = cv2.erode(binary, erode_kernel, iterations=1)
        
        if dilate_radius > 0:
            dilate_kernel = np.ones((dilate_radius * 2 + 1, dilate_radius * 2 + 1), np.uint8)
            binary = cv2.dilate(binary, dilate_kernel, iterations=1)
        
        # Create trimap by eroding and dilating the binary mask
        if unknown_width > 0:
            kernel = np.ones((unknown_width, unknown_width), np.uint8)
            
            # Definite foreground: eroded mask
            fg = cv2.erode(binary, kernel, iterations=1)
            
            # Definite background: inverted dilated mask
            dilated = cv2.dilate(binary, kernel, iterations=1)
            bg = 255 - dilated
            
            # Build trimap
            trimap = np.full_like(binary, 128)  # Start with unknown
            trimap[fg == 255] = 255  # Definite FG
            trimap[bg == 255] = 0    # Definite BG
        else:
            # No unknown region - use binary mask directly
            trimap = binary
        
        return trimap
    
    def refine(
        self,
        image: np.ndarray,
        coarse_mask: np.ndarray,
        trimap_width: int = 10,
        erode_radius: int = 2,
        dilate_radius: int = 2
    ) -> np.ndarray:
        """
        Refine a coarse mask using ViTMatte.
        
        Args:
            image: RGB image (HxWx3, uint8)
            coarse_mask: Coarse segmentation mask (HxW, 0-1 float)
            trimap_width: Width of uncertain region in trimap
            erode_radius: Trimap erosion cleanup radius
            dilate_radius: Trimap dilation cleanup radius
        
        Returns:
            High-quality alpha matte (HxW, 0-1 float)
        """
        import torch
        
        self.load()
        
        # Create trimap
        trimap = self.create_trimap(
            coarse_mask,
            unknown_width=trimap_width,
            erode_radius=erode_radius,
            dilate_radius=dilate_radius
        )
        
        # Ensure image is uint8 RGB
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Process inputs
        inputs = self.processor(
            images=image,
            trimaps=trimap,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            alpha = outputs.alphas[0, 0].cpu().numpy()
        
        # Resize back to original size if needed
        if alpha.shape != coarse_mask.shape:
            alpha = cv2.resize(
                alpha,
                (coarse_mask.shape[1], coarse_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        return alpha.astype(np.float32)
    
    def refine_with_crop(
        self,
        image: np.ndarray,
        coarse_mask: np.ndarray,
        padding_percent: float = 20.0,
        trimap_width: int = 10,
        erode_radius: int = 2,
        dilate_radius: int = 2
    ) -> np.ndarray:
        """
        Refine mask with automatic cropping for efficiency.
        
        Crops to the bounding box of the mask plus padding,
        runs refinement, then places back into full resolution.
        """
        # Find bounding box of mask
        mask_binary = (coarse_mask > 0.5).astype(np.uint8)
        coords = np.where(mask_binary > 0)
        
        if len(coords[0]) == 0:
            # Empty mask
            return coarse_mask
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Add padding
        h, w = coarse_mask.shape[:2]
        pad_y = int((y_max - y_min) * padding_percent / 100)
        pad_x = int((x_max - x_min) * padding_percent / 100)
        
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)
        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        
        # Crop
        image_crop = image[y_min:y_max, x_min:x_max]
        mask_crop = coarse_mask[y_min:y_max, x_min:x_max]
        
        # Refine cropped region
        alpha_crop = self.refine(
            image_crop,
            mask_crop,
            trimap_width=trimap_width,
            erode_radius=erode_radius,
            dilate_radius=dilate_radius
        )
        
        # Place back into full resolution
        alpha_full = np.zeros_like(coarse_mask, dtype=np.float32)
        alpha_full[y_min:y_max, x_min:x_max] = alpha_crop
        
        return alpha_full
    
    def unload(self):
        """Unload model to free memory."""
        import torch
        
        self.model = None
        self.processor = None
        self._loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[ViTMatte] Model unloaded.")


# Global refiner instance
_refiner: Optional[ViTMatteRefiner] = None


def get_refiner() -> ViTMatteRefiner:
    """Get or create the global ViTMatte refiner."""
    global _refiner
    if _refiner is None:
        _refiner = ViTMatteRefiner()
    return _refiner


def refine_mask(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    trimap_width: int = 10,
    use_crop: bool = True,
    crop_padding: float = 20.0
) -> np.ndarray:
    """
    Convenience function to refine a mask using ViTMatte.
    
    Args:
        image: RGB image
        coarse_mask: Coarse segmentation mask
        trimap_width: Width of uncertain region
        use_crop: Whether to crop for efficiency
        crop_padding: Padding percentage when cropping
    
    Returns:
        Refined alpha matte
    """
    refiner = get_refiner()
    
    if use_crop:
        return refiner.refine_with_crop(
            image, coarse_mask,
            padding_percent=crop_padding,
            trimap_width=trimap_width
        )
    else:
        return refiner.refine(
            image, coarse_mask,
            trimap_width=trimap_width
        )
