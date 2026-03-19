# matanyone2_refiner.py — Alpha matte refinement using MatAnyone2
# Replaces vitmatte_refiner.py for higher-quality soft alpha mattes.
#
# MatAnyone2 takes an RGB frame + coarse binary mask and produces
# a production-quality alpha matte with natural edge transitions.
#
# For single frames, the "video" matting model processes one frame.
# For sequences, it leverages temporal memory for consistency.

import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import threading

PACKAGE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

_model = None
_cfg = None
_device = None
_lock = threading.Lock()


def _ensure_imports():
    """Ensure MatAnyone2 package is importable."""
    tp = str(PACKAGE_DIR / "third_party" / "MatAnyone2")
    if tp not in sys.path:
        sys.path.insert(0, tp)


def load(device=None):
    """Load MatAnyone2 model (lazy, thread-safe)."""
    global _model, _cfg, _device

    if _model is not None:
        return

    with _lock:
        if _model is not None:
            return

        from . import model_manager
        _model, _cfg = model_manager.build_matanyone2(device=device)
        _device = next(_model.parameters()).device

        print("[MatAnyone2] Refiner loaded.")


def unload():
    """Unload model to free GPU memory."""
    global _model, _cfg, _device
    import torch

    with _lock:
        _model = None
        _cfg = None
        _device = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[MatAnyone2] Refiner unloaded.")


def refine_single_frame(
    image: np.ndarray,
    coarse_mask: np.ndarray,
) -> np.ndarray:
    """Refine a coarse binary mask into a soft alpha matte (single frame).

    Args:
        image: RGB uint8 (H, W, 3).
        coarse_mask: Binary mask from SAM, (H, W) float [0, 1].

    Returns:
        Alpha matte (H, W) float32 [0, 1].
    """
    import torch

    _ensure_imports()
    load()

    from matanyone2.inference.inference_core import InferenceCore

    processor = InferenceCore(_model, _cfg, device=_device)

    # Convert image to (3, H, W) float32 [0, 255]
    if image.dtype == np.uint8:
        img_f = image.astype(np.float32)
    else:
        img_f = (image * 255.0).astype(np.float32)

    img_tensor = torch.from_numpy(img_f).permute(2, 0, 1).to(_device)

    # Convert mask to (1, H, W) uint8
    mask_uint8 = (coarse_mask > 0.5).astype(np.uint8)
    mask_tensor = torch.from_numpy(mask_uint8).unsqueeze(0).to(_device)

    with torch.inference_mode():
        output_prob = processor.step(
            image=img_tensor,
            mask=mask_tensor,
            objects=[1],
            first_frame_pred=True,
            matting=True,
        )
        alpha = processor.output_prob_to_mask(output_prob, matting=True)

    alpha_np = alpha.cpu().float().numpy()
    while alpha_np.ndim > 2:
        alpha_np = alpha_np.squeeze(0)

    return np.clip(alpha_np, 0.0, 1.0).astype(np.float32)


def refine_sequence(
    images: List[np.ndarray],
    first_mask: np.ndarray,
    mem_every: int = 5,
    max_mem_frames: int = 5,
    progress_callback=None,
) -> List[np.ndarray]:
    """Refine a sequence of frames using MatAnyone2 temporal matting.

    Args:
        images: List of RGB uint8 (H, W, 3) frames.
        first_mask: Coarse binary mask for the first frame, (H, W) float [0, 1].
        mem_every: Update memory every N frames.
        max_mem_frames: Max frames in long-term memory.
        progress_callback: Optional callable(frame_idx, total_frames).

    Returns:
        List of alpha mattes, each (H, W) float32 [0, 1].
    """
    import torch
    from omegaconf import OmegaConf

    _ensure_imports()
    load()

    from matanyone2.inference.inference_core import InferenceCore

    # Build a per-sequence config with user settings
    seq_cfg = OmegaConf.create({
        "mem_every": mem_every,
        "max_mem_frames": max_mem_frames,
        "chunk_size": -1,
        "max_internal_size": -1,
        "stagger_updates": mem_every,
        "top_k": 30,
    })

    processor = InferenceCore(_model, seq_cfg, device=_device)
    alphas = []
    total = len(images)

    with torch.inference_mode():
        for i, img in enumerate(images):
            if img.dtype == np.uint8:
                img_f = img.astype(np.float32)
            else:
                img_f = (img * 255.0).astype(np.float32)

            img_tensor = torch.from_numpy(img_f).permute(2, 0, 1).to(_device)

            if i == 0:
                mask_uint8 = (first_mask > 0.5).astype(np.uint8)
                mask_tensor = torch.from_numpy(mask_uint8).unsqueeze(0).to(_device)
                output_prob = processor.step(
                    image=img_tensor,
                    mask=mask_tensor,
                    objects=[1],
                    first_frame_pred=True,
                    matting=True,
                )
            else:
                output_prob = processor.step(
                    image=img_tensor,
                    matting=True,
                )

            alpha = processor.output_prob_to_mask(output_prob, matting=True)
            alpha_np = alpha.cpu().float().numpy()
            while alpha_np.ndim > 2:
                alpha_np = alpha_np.squeeze(0)

            alphas.append(np.clip(alpha_np, 0.0, 1.0).astype(np.float32))

            if progress_callback:
                progress_callback(i, total)

    return alphas
