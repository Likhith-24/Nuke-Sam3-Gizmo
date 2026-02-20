# vitmatte_refiner.py - High-quality alpha matte refinement using ViTMatte
# Converts SAM coarse masks into production-quality alpha mattes.
# Model weights are cached locally inside models/vitmatte/ for portability.

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import cv2

PACKAGE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = PACKAGE_DIR / "models"
VITMATTE_CACHE = MODELS_DIR / "vitmatte"

_MODEL_NAME = "hustvl/vitmatte-small-composition-1k"


class ViTMatteRefiner:
    """ViTMatte-based alpha matte refinement for high-quality edges."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self._loaded = False
    
    def load(self):
        """Load the ViTMatte model.

        Weights are downloaded to ``models/vitmatte/`` on the first call
        and loaded from there on subsequent calls — no reliance on the
        default ``~/.cache/huggingface/`` directory.
        """
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

        # Ensure local cache directory exists
        VITMATTE_CACHE.mkdir(parents=True, exist_ok=True)

        cache_dir = str(VITMATTE_CACHE)
        self.processor = VitMatteImageProcessor.from_pretrained(
            _MODEL_NAME, cache_dir=cache_dir,
        )
        self.model = VitMatteForImageMatting.from_pretrained(
            _MODEL_NAME, cache_dir=cache_dir,
        )
        self.model.to(self.device)
        self.model.eval()
        
        self._loaded = True
        print("[ViTMatte] Model loaded successfully.")
    
    def create_trimap(
        self,
        mask: np.ndarray,
        erode_radius: int = 5,
        dilate_radius: int = 15,
    ) -> np.ndarray:
        """Create a trimap from a binary/soft mask for ViTMatte.

        The trimap defines three regions:
          - **Definite FG** (255): mask eroded inward by *erode_radius*.
          - **Definite BG** (0):   outside the mask dilated by *dilate_radius*.
          - **Unknown**    (128): the transition band between them.

        The unknown band width = erode_radius + dilate_radius, centred on
        SAM's original boundary.  ViTMatte only refines the unknown zone,
        producing natural soft edges (hair, fur, glass, translucency).

        Args:
            mask: Binary mask from SAM (0-1 float or 0-255 uint8).
            erode_radius: Pixels *inward* from SAM boundary → inner edge
                          softness.  Larger = more of the interior is
                          re-evaluated by ViTMatte.
            dilate_radius: Pixels *outward* from SAM boundary → outer edge
                           reach.  Increase for wispy hair / fur / smoke.

        Returns:
            Trimap (H×W uint8): 0 = BG, 128 = unknown, 255 = FG.
        """
        # Normalize to uint8 binary
        if mask.dtype in (np.float32, np.float64):
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.astype(np.uint8)
        _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

        # Elliptical kernels produce smooth, natural trimap boundaries
        # (square kernels cause blocky artefacts on curved edges).
        if erode_radius > 0:
            e_kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (erode_radius * 2 + 1, erode_radius * 2 + 1),
            )
            fg = cv2.erode(binary, e_kern, iterations=1)
        else:
            fg = binary.copy()

        if dilate_radius > 0:
            d_kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilate_radius * 2 + 1, dilate_radius * 2 + 1),
            )
            dilated = cv2.dilate(binary, d_kern, iterations=1)
        else:
            dilated = binary.copy()

        # Build trimap: start as BG, mark dilated band as unknown,
        # then overwrite the eroded core as definite FG.
        trimap = np.zeros_like(binary, dtype=np.uint8)
        trimap[dilated > 0] = 128
        trimap[fg > 0] = 255

        return trimap
    
    def refine(
        self,
        image: np.ndarray,
        coarse_mask: np.ndarray,
        erode_radius: int = 5,
        dilate_radius: int = 15,
    ) -> np.ndarray:
        """Refine a coarse mask using ViTMatte for production-quality alpha.

        Args:
            image: RGB image (HxWx3, uint8).
            coarse_mask: Coarse binary mask from SAM (HxW, 0-1 float).
            erode_radius: Inner edge softness (px inward from SAM boundary).
            dilate_radius: Outer edge reach (px outward — hair/fur/glass).

        Returns:
            Soft alpha matte (HxW, 0-1 float) with natural edge transitions.
        """
        import torch

        self.load()

        # Create trimap — the unknown band is where ViTMatte works its magic
        trimap = self.create_trimap(
            coarse_mask,
            erode_radius=erode_radius,
            dilate_radius=dilate_radius,
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
        
        # Run inference.
        # SAM3's Sam3TrackerPredictor.__init__() enters a PERSISTENT
        # bfloat16 autocast context that never exits.  This leaks into
        # every subsequent torch operation — including ViTMatte.
        # We explicitly disable autocast here so ViTMatte runs in its
        # native float32 precision, then cast the output to float32
        # before converting to numpy (which doesn't support bf16).
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad(), torch.autocast(device_type=_dev, enabled=False):
            # Re-cast inputs to float32 in case autocast already
            # converted them before we disabled it.
            inputs = {
                k: v.float() if v.is_floating_point() else v
                for k, v in inputs.items()
            }
            outputs = self.model(**inputs)
            alpha = outputs.alphas[0, 0].cpu().float().numpy()
        
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
        erode_radius: int = 5,
        dilate_radius: int = 15,
    ) -> np.ndarray:
        """Refine mask with automatic cropping for efficiency and quality.

        Crops to the mask bounding-box + *padding_percent* so ViTMatte
        gets more detail resolution on the actual edges rather than
        wasting capacity on irrelevant background areas.
        """
        # Find bounding box of mask
        mask_binary = (coarse_mask > 0.5).astype(np.uint8)
        coords = np.where(mask_binary > 0)

        if len(coords[0]) == 0:
            return coarse_mask  # empty mask — nothing to refine

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # Add padding (ensure the trimap's dilate band fits in the crop)
        h, w = coarse_mask.shape[:2]
        pad_y = max(int((y_max - y_min) * padding_percent / 100), dilate_radius + 4)
        pad_x = max(int((x_max - x_min) * padding_percent / 100), dilate_radius + 4)

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
            erode_radius=erode_radius,
            dilate_radius=dilate_radius,
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
    erode_radius: int = 5,
    dilate_radius: int = 15,
    crop_padding: float = 20.0,
) -> np.ndarray:
    """Convenience function to refine a mask using ViTMatte.

    Always uses cropped refinement for efficiency and quality.
    """
    refiner = get_refiner()
    return refiner.refine_with_crop(
        image, coarse_mask,
        padding_percent=crop_padding,
        erode_radius=erode_radius,
        dilate_radius=dilate_radius,
    )
