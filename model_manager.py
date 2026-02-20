"""model_manager.py – SAM model registry, download helper, and builder.

Supports SAM 2.0 / 2.1 (Tiny · Small · Base+ · Large) and SAM 3.
Checkpoints are stored in the ``models/`` directory next to this file.
Config YAMLs are shipped locally in ``configs/`` — no reliance on the
sam2 pip package's Hydra search paths.
All models are combined with ViTMatte for production-quality alpha mattes.
"""

import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional

PACKAGE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = PACKAGE_DIR / "models"
CONFIGS_DIR = PACKAGE_DIR / "configs"

# ────────────────────────────────────────────────────────────────────
#  Model registry
# ────────────────────────────────────────────────────────────────────
# Keys: (version, size)  →  cfg, file, url, mb

_SAM2: Dict[tuple, Dict[str, Any]] = {
    # ── SAM 2.0 ──
    ("2.0", "Tiny"): dict(
        cfg="sam2_hiera_t.yaml",
        file="sam2_hiera_tiny.pt",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        mb=156,
    ),
    ("2.0", "Small"): dict(
        cfg="sam2_hiera_s.yaml",
        file="sam2_hiera_small.pt",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        mb=185,
    ),
    ("2.0", "Base+"): dict(
        cfg="sam2_hiera_b+.yaml",
        file="sam2_hiera_base_plus.pt",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        mb=324,
    ),
    ("2.0", "Large"): dict(
        cfg="sam2_hiera_l.yaml",
        file="sam2_hiera_large.pt",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        mb=898,
    ),
    # ── SAM 2.1 ──
    ("2.1", "Tiny"): dict(
        cfg="sam2.1_hiera_t.yaml",
        file="sam2.1_hiera_tiny.pt",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        mb=156,
    ),
    ("2.1", "Small"): dict(
        cfg="sam2.1_hiera_s.yaml",
        file="sam2.1_hiera_small.pt",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        mb=185,
    ),
    ("2.1", "Base+"): dict(
        cfg="sam2.1_hiera_b+.yaml",
        file="sam2.1_hiera_base_plus.pt",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        mb=324,
    ),
    ("2.1", "Large"): dict(
        cfg="sam2.1_hiera_l.yaml",
        file="sam2.1_hiera_large.pt",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        mb=898,
    ),
}

_SAM3: Dict[str, Any] = dict(
    file="sam3.pt",
    url="hf://facebook/sam3",  # downloaded via huggingface_hub
    mb=3290,
)

# Precision dtype map
_DTYPE = {"fp32": None, "fp16": "float16", "bf16": "bfloat16"}


# ────────────────────────────────────────────────────────────────────
#  Lookup helpers
# ────────────────────────────────────────────────────────────────────

def get_info(family: str, version: str = "2.1", size: str = "Large") -> Dict[str, Any]:
    """Return the registry entry for the given model selection."""
    if family == "SAM2":
        key = (version, size)
        if key not in _SAM2:
            raise KeyError(f"Unknown SAM2 variant: version={version}, size={size}")
        return _SAM2[key]
    return _SAM3


def checkpoint_path(family: str, version: str = "2.1", size: str = "Large") -> Path:
    """Return the expected local path of the checkpoint file."""
    return MODELS_DIR / get_info(family, version, size)["file"]


def is_downloaded(family: str, version: str = "2.1", size: str = "Large") -> bool:
    """Check whether the checkpoint exists on disk."""
    return checkpoint_path(family, version, size).exists()


def status_text(family: str, version: str = "2.1", size: str = "Large") -> str:
    """Human-readable model status string for the UI label."""
    cp = checkpoint_path(family, version, size)
    if cp.exists():
        mb = cp.stat().st_size / (1024 ** 2)
        return f"Ready  ({cp.name}, {mb:.0f} MB)"
    info = get_info(family, version, size)
    if info.get("url"):
        return f"Not downloaded  (~{info['mb']} MB)"
    return "Not downloaded  (place checkpoint manually)"


# ────────────────────────────────────────────────────────────────────
#  Download
# ────────────────────────────────────────────────────────────────────

def _progress_hook(block_num: int, block_size: int, total_size: int):
    done = block_num * block_size
    if total_size > 0:
        pct = min(done / total_size * 100, 100)
        sys.stdout.write(
            f"\r[H2 SamViT] Downloading… {pct:5.1f}%  "
            f"({done / 1048576:.0f}/{total_size / 1048576:.0f} MB)"
        )
    else:
        sys.stdout.write(f"\r[H2 SamViT] Downloading… {done / 1048576:.0f} MB")
    sys.stdout.flush()


def download(family: str, version: str = "2.1", size: str = "Large") -> str:
    """Download a checkpoint.  Returns the local path on success."""
    cp = checkpoint_path(family, version, size)
    if cp.exists():
        print(f"[H2 SamViT] Checkpoint already present: {cp}")
        return str(cp)

    info = get_info(family, version, size)
    url = info.get("url", "")
    if not url:
        raise RuntimeError(
            f"No download URL for {family} (v{version}, {size}).\n"
            f"Place the checkpoint manually in:\n  {MODELS_DIR}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── HuggingFace download (SAM3) ──
    if url.startswith("hf://"):
        repo_id = url[len("hf://"):]       # e.g. "facebook/sam3"
        print(f"[H2 SamViT] Downloading {cp.name} from HuggingFace ({repo_id}) …")
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id=repo_id, filename=info["file"])
            import shutil
            shutil.copy2(local, str(cp))
            print(f"[H2 SamViT] Saved → {cp}")
        except Exception:
            try:
                cp.unlink(missing_ok=True)
            except Exception:
                pass
            raise
        return str(cp)

    # ── Regular HTTP download (SAM2) ──
    tmp = str(cp) + ".part"
    print(f"[H2 SamViT] Downloading {cp.name} …")
    print(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress_hook)
        print()  # newline after progress
        os.rename(tmp, str(cp))
        print(f"[H2 SamViT] Saved → {cp}")
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise

    return str(cp)


# ────────────────────────────────────────────────────────────────────
#  Model builders
# ────────────────────────────────────────────────────────────────────

def _get_device():
    """Return the best available torch device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _torch_dtype(precision: str):
    """Return a ``torch.dtype`` or *None* for the given precision tag."""
    import torch
    name = _DTYPE.get(precision)
    return getattr(torch, name) if name else None


def _build_sam2_from_local_config(cfg_name: str, ckpt_path: str, device):
    """Build a SAM2Base model from a local YAML config + checkpoint.

    Loads the config from our shipped ``configs/`` directory using
    ``omegaconf`` + ``hydra.utils.instantiate`` directly — no reliance
    on Hydra's config-module search path inside the sam2 pip package.
    """
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    cfg_path = CONFIGS_DIR / cfg_name
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found: {cfg_path}\n"
            f"Re-install the plugin or restore the configs/ directory."
        )

    cfg = OmegaConf.load(str(cfg_path))
    OmegaConf.resolve(cfg)

    OmegaConf.update(cfg, "model.sam_mask_decoder_extra_args", {
        "dynamic_multimask_via_stability": True,
        "dynamic_multimask_stability_delta": 0.05,
        "dynamic_multimask_stability_thresh": 0.98,
    }, merge=True)

    model = instantiate(cfg.model, _recursive_=True)

    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)

    model = model.to(device)
    model.eval()
    return model


def build_sam2_predictor(
    version: str = "2.1",
    size: str = "Large",
    precision: str = "fp16",
    device=None,
):
    """Build a ``SAM2ImagePredictor`` for the given SAM 2.x variant.

    Loads the config YAML from ``configs/`` and the checkpoint from
    ``models/``.  Auto-downloads the checkpoint if missing.
    """
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = device or _get_device()

    cp = checkpoint_path("SAM2", version, size)
    if not cp.exists():
        download("SAM2", version, size)

    info = get_info("SAM2", version, size)
    cfg = info["cfg"]

    print(f"[H2 SamViT] Loading SAM2 v{version} {size} on {device}  "
          f"(precision={precision}) …")

    model = _build_sam2_from_local_config(cfg, str(cp), device)

    # IMPORTANT: Do NOT cast model weights to fp16/bf16!
    # Like SAM3, SAM2 must stay in float32.  Mixed precision is handled
    # dynamically via torch.autocast during inference — this is the
    # official inference pattern from facebookresearch/sam2.
    # Casting weights directly causes numerical precision loss in the
    # mask decoder and degrades segmentation quality.

    # Enable TF32 for Ampere+ GPUs (faster fp32 math, no accuracy loss)
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    predictor = SAM2ImagePredictor(model)
    print(f"[H2 SamViT] SAM2 v{version} {size} ready.")
    return predictor


def build_sam3_model(
    precision: str = "fp16",
    device=None,
    enable_inst_interactivity: bool = True,
):
    """Build the SAM 3 image model from the local checkpoint.

    SAM 3 is a completely different architecture from SAM 2 — it uses
    a unified detector (vision + language backbone, geometry encoder,
    transformer, segmentation head) plus a SAM2-like tracker for
    interactive point/box prompts.

    Returns the ``Sam3Image`` model.  Wrap it in ``Sam3Processor``
    for text prompts or access ``.inst_interactive_predictor`` for
    SAM1-style point/box prediction.
    """
    import torch
    from sam3.model_builder import build_sam3_image_model

    device = device or _get_device()

    cp = checkpoint_path("SAM3")
    if not cp.exists():
        download("SAM3")

    print(f"[H2 SamViT] Loading SAM3 on {device}  (precision={precision}) …")

    model = build_sam3_image_model(
        checkpoint_path=str(cp),
        load_from_HF=False,
        device=str(device),
        enable_inst_interactivity=enable_inst_interactivity,
        enable_segmentation=True,
        eval_mode=True,
    )

    # Share the detector's vision-language backbone with the tracker.
    # build_tracker() creates the tracker WITHOUT a backbone (with_backbone=False)
    # because in video mode the detector feeds features to the tracker.
    # For standalone interactive image prediction (SAM1-style), the tracker
    # needs its own forward_image path — satisfied by sharing the detector's
    # SAM3VLBackbone.
    if enable_inst_interactivity and model.inst_interactive_predictor is not None:
        model.inst_interactive_predictor.model.backbone = model.backbone

    # IMPORTANT: Do NOT cast model weights to fp16/bf16!
    # SAM3 must stay in float32.  Mixed precision is handled at the
    # activation level via torch.autocast during inference — the same
    # approach used by ComfyUI-SAM3 / the official codebase.
    # Casting the weights directly causes dtype mismatches in the
    # decoder's FFN layers (some internal tensors are created in fp32).

    # Enable TF32 for Ampere+ GPUs (faster fp32 math, no accuracy loss)
    if device.type == "cuda":
        import torch
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    print("[H2 SamViT] SAM3 ready.")
    return model
