# inference.py — SAM3/SAM2 segmentation + MatAnyone2 alpha matting engine
#
# Pipeline:
#   1. Nuke renders the current frame to a temp PNG (in-process)
#   2. worker.py — a subprocess on the plugin venv's Python — runs SAM +
#      MatAnyone2 + mask pre/post processing and writes the mask PNG
#   3. Temporal consistency (numpy, in-process)
#   4. Mask written to the gizmo's internal Read node (in-process)
#
# WHY A SUBPROCESS: Nuke bundles its own libtorch (c10.dll / torch_cpu.dll,
# for the Inference & CopyCat nodes), which is already resident in the
# process. Importing pip-torch on top of it fails with WinError 1114, so
# torch must never be imported inside Nuke. numpy / cv2 / PIL are safe.
#
# The nuke-free helpers in this module (finalize_mask, preprocess_mask,
# postprocess_mask, load_text_model, select_detection_by_point,
# _resolve_autocast_ctx) are imported and reused BY the worker — keep them
# free of `import nuke` at module or function level.

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import threading


def _ensure_packages():
    """
    Pre-flight check — abort with a helpful dialog if any ML package
    is missing (i.e. the user hasn't run install.py yet).
    """
    from . import env_bootstrap
    if env_bootstrap.is_ready():
        return
    # Not ready — show a message inside Nuke
    try:
        import nuke
        nuke.message(
            "H2 SamViT — packages not installed\n\n"
            + env_bootstrap.get_status_message()
        )
    except ImportError:
        pass
    raise RuntimeError(
        "[H2 SamViT] Required ML packages are missing. "
        "Run install.py from a terminal first."
    )


# Global model instances — only ever populated when this module runs INSIDE
# the worker process (load_sam_model/load_text_model must not be called from
# Nuke; see module docstring).
_sam_predictor = None         # SAM2ImagePredictor  or  SAM3InteractiveImagePredictor
_sam3_model = None            # Full Sam3Image model (needed for Sam3Processor text path)
_current_model_key = None     # (family, version, size, precision) – reload on change
_text_model = None            # Grounding DINO (SAM2 text-prompt path only)
_model_lock = threading.Lock()


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_autocast_ctx(precision: str):
    """Return a ``torch.autocast`` context for the given precision knob.

    Maps the user-facing precision string (``fp16`` / ``bf16`` / ``fp32``)
    to the correct ``torch.autocast`` call.  Falls back gracefully when
    the GPU does not support the requested dtype.

    Both SAM2 and SAM3 keep their weights in float32 — autocast handles
    mixed-precision dynamically at the activation level, following the
    official inference pattern from facebookresearch/sam2 and sam3.
    """
    import torch
    from contextlib import nullcontext

    if precision == "fp32" or not torch.cuda.is_available():
        return nullcontext()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(precision, torch.bfloat16)

    # Fall back to fp16 if GPU doesn't support bf16 (pre-Ampere)
    if dtype == torch.bfloat16:
        try:
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                dtype = torch.float16
        except Exception:
            dtype = torch.float16

    return torch.autocast(device_type="cuda", dtype=dtype)


def load_sam_model(node):
    """Load the SAM model selected in the node's knobs.

    WARNING: imports torch — must only run inside the worker process. Kept
    for legacy callers (callbacks_v3.process_sequence); the main entry
    points below no longer use it in-process.
    """
    global _sam_predictor, _sam3_model, _current_model_key
    _ensure_packages()

    import nuke
    family    = node.knob("model_family").value()
    precision = node.knob("model_precision").value()

    # SAM3 ignores the version/size knobs (they're hidden in the UI)
    if family == "SAM3":
        version = "3.0"
        size    = "Default"
    else:
        version = node.knob("sam_version").value()
        size    = node.knob("model_size").value()

    key = (family, version, size, precision)

    if _sam_predictor is not None and _current_model_key == key:
        return _sam_predictor

    with _model_lock:
        # Double-check after acquiring lock
        if _sam_predictor is not None and _current_model_key == key:
            return _sam_predictor

        # Free previous model
        _free_models()

        from . import model_manager

        # Prompt download if checkpoint is missing
        if not model_manager.is_downloaded(family, version, size):
            info = model_manager.get_info(family, version, size)
            url  = info.get("url", "")
            if not url:
                raise FileNotFoundError(
                    "Checkpoint not found and no download URL is configured.\n"
                    f"Place it manually in:\n  {model_manager.MODELS_DIR}"
                )
            label = f"{family} v{version} {size}" if family == "SAM2" else "SAM3"
            if not nuke.ask(
                f"{label} checkpoint not found.\n\n"
                f"Download now?  (~{info['mb']} MB)\n"
                "Progress will be printed to the Script Editor."
            ):
                raise RuntimeError("Download cancelled by user.")

        # ── Build the right predictor ──
        if family == "SAM3":
            _sam3_model = model_manager.build_sam3_model(
                precision=precision,
            )
            _sam_predictor = _sam3_model.inst_interactive_predictor
        else:
            _sam_predictor = model_manager.build_sam2_predictor(
                version=version, size=size, precision=precision,
            )

        _current_model_key = key

        # Update status label on the node
        try:
            node.knob("model_status").setValue(
                model_manager.status_text(family, version, size)
            )
        except Exception:
            pass

        return _sam_predictor


def _free_models():
    """Release all loaded models and free GPU memory."""
    global _sam_predictor, _sam3_model, _current_model_key
    _sam_predictor = None
    _sam3_model = None
    _current_model_key = None
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_text_model():
    """Load text-to-detection model (Grounding DINO style).

    WARNING: imports torch — worker-process only.

    Weights are downloaded once to ``models/grounding_dino/`` and
    loaded locally on subsequent calls — no dependency on network
    access after the initial download.
    """
    global _text_model
    _ensure_packages()

    if _text_model is not None:
        return _text_model

    with _model_lock:
        if _text_model is not None:
            return _text_model

        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from pathlib import Path

        device = get_device()
        print(f"[H2 SamViT] Loading text detection model on {device}...")

        model_name = "IDEA-Research/grounding-dino-base"
        cache_dir = str(
            Path(os.path.dirname(os.path.abspath(__file__)))
            / "models" / "grounding_dino"
        )
        os.makedirs(cache_dir, exist_ok=True)

        # Try local-only first (offline), fall back to download
        try:
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_name, cache_dir=cache_dir, local_files_only=True,
            )
            processor = AutoProcessor.from_pretrained(
                model_name, cache_dir=cache_dir, local_files_only=True,
            )
        except Exception:
            print("[H2 SamViT] Downloading Grounding DINO (first time)…")
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_name, cache_dir=cache_dir,
            )
            processor = AutoProcessor.from_pretrained(
                model_name, cache_dir=cache_dir,
            )

        _text_model = {
            "model": model.to(device),
            "processor": processor,
            "device": device,
        }

        _text_model["model"].eval()

        print("[H2 SamViT] Text detection model loaded successfully.")
        return _text_model


def render_frame_to_png(node) -> str:
    """Render the node's input at the current frame to a temp PNG.

    Returns the file path — the CALLER is responsible for deleting it.
    Renders through a temporary Write node so every format Nuke can decode
    is supported and upstream colour/transform ops are baked in.
    """
    import nuke
    import tempfile

    input_node = node.input(0)
    if not input_node:
        raise ValueError("No input connected to H2_SamViT node.")

    frame = nuke.frame()
    # Forward slashes: Nuke file knobs mangle Windows backslashes (\t, \a …
    # are treated as escapes), silently writing to a wrong path.
    tmp = os.path.join(
        tempfile.gettempdir(),
        f"_h2samvit_input_{os.getpid()}_{frame}.png",
    ).replace("\\", "/")

    write = nuke.nodes.Write()
    write["file"].setValue(tmp)
    write["file_type"].setValue("png")
    write.setInput(0, input_node)
    try:
        nuke.execute(write, frame, frame)
    finally:
        nuke.delete(write)

    if not os.path.exists(tmp):
        raise RuntimeError(f"Could not render the input frame to {tmp}")
    print(f"[H2 SamViT] Input captured — frame {frame}")
    return tmp


def image_from_nuke_node(node) -> np.ndarray:
    """Extract the current frame from the node's input as a uint8 RGB array."""
    import cv2

    tmp = render_frame_to_png(node)
    try:
        bgr = cv2.imread(tmp, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read the rendered frame from {tmp}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def refine_mask_with_matanyone2(
    image: np.ndarray,
    coarse_mask: np.ndarray,
) -> np.ndarray:
    """Refine a coarse mask using MatAnyone2 for production-quality alpha."""
    from . import matanyone2_refiner
    return matanyone2_refiner.refine_single_frame(image, coarse_mask)


def finalize_mask(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    params: Dict[str, Any],
    use_ma2: bool,
) -> Tuple[np.ndarray, bool]:
    """Shared, nuke-free mask pipeline (runs inside the worker).

    preprocess → optional MatAnyone2 soft-alpha refinement (else pure
    binary) → postprocess.  Returns ``(mask, soft)`` where ``soft`` is True
    when the mask is a soft alpha matte rather than binary.
    """
    while coarse_mask.ndim > 2:
        coarse_mask = coarse_mask.squeeze(0)

    coarse_mask = preprocess_mask(coarse_mask, params)

    if use_ma2:
        alpha = refine_mask_with_matanyone2(image, coarse_mask)

        # Normalize: ensure the FG core reaches 1.0
        alpha_max = float(alpha.max())
        if 0.01 < alpha_max < 0.95:
            alpha = np.clip(alpha / alpha_max, 0.0, 1.0)
        else:
            alpha = np.clip(alpha, 0.0, 1.0)

        # Clean near-zero noise in definite BG areas
        alpha[alpha < 0.004] = 0.0

        # MatAnyone2 produces proper soft alpha — don't binarize
        params = dict(params)
        params["final_binary"] = False
        print("[H2 SamViT] MatAnyone2 refinement applied")
        soft = True
    else:
        # Pure binary mask — strictly 0.0 or 1.0
        alpha = (coarse_mask > 0.5).astype(np.float32)
        soft = False

    return postprocess_mask(alpha, params), soft


# ──────────────────────────────────────────────────────────────────────
#  Worker-routed entry points (called from the gizmo's buttons)
# ──────────────────────────────────────────────────────────────────────

def model_ready(node) -> Tuple[bool, str]:
    """Cheap pre-flight for the inference buttons. Returns (ready, reason).

    Checked BEFORE Run Inference / Process Sequence does any real work, so a
    missing environment or checkpoint refuses immediately with a clear
    message instead of rendering the frame, spawning the worker and failing
    (or surprise-downloading a multi-GB model) mid-flight.
    """
    from . import env_bootstrap, model_manager

    venv = env_bootstrap.get_venv_path()
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        return False, (
            "The ML environment is not installed.\n\n"
            f"Expected venv:\n  {venv}\n\n"
            "Run install.py from a terminal first (see README)."
        )

    family = node.knob("model_family").value()
    if family == "SEC-4B":
        return False, (
            "SeC-4B inference is not yet integrated.\n"
            "Please use SAM2 or SAM3."
        )
    if family == "SAM3":
        version, size = "3.0", "Default"
    else:
        version = node.knob("sam_version").value()
        size = node.knob("model_size").value()

    if not model_manager.is_downloaded(family, version, size):
        info = model_manager.get_info(family, version, size)
        label = f"{family} v{version} {size}" if family == "SAM2" else family
        return False, (
            f"{label} checkpoint is not downloaded yet "
            f"(~{info.get('mb', '?')} MB).\n\n"
            "Click 'Download Model' on the node first, then run inference."
        )

    return True, ""


def _get_model_request(node) -> Dict[str, Any]:
    """Read the model knobs; confirm a checkpoint download if needed."""
    import nuke
    from . import model_manager

    family    = node.knob("model_family").value()
    precision = node.knob("model_precision").value()
    if family == "SAM3":
        version, size = "3.0", "Default"
    else:
        version = node.knob("sam_version").value()
        size    = node.knob("model_size").value()

    allow_download = False
    if not model_manager.is_downloaded(family, version, size):
        info = model_manager.get_info(family, version, size)
        if not info.get("url"):
            raise FileNotFoundError(
                "Checkpoint not found and no download URL is configured.\n"
                f"Place it manually in:\n  {model_manager.MODELS_DIR}"
            )
        label = f"{family} v{version} {size}" if family == "SAM2" else family
        if not nuke.ask(
            f"{label} checkpoint not found.\n\n"
            f"Download now?  (~{info['mb']} MB)\n"
            "Progress will be printed to the terminal."
        ):
            raise RuntimeError("Download cancelled by user.")
        allow_download = True

    return {
        "family": family,
        "version": version,
        "size": size,
        "precision": precision,
        "allow_download": allow_download,
    }


def _use_ma2(node) -> bool:
    """Read the refinement toggle (knob name varies across gizmo versions)."""
    for name in ("use_matanyone2", "use_vitmatte"):
        k = node.knob(name)
        if k is not None:
            return bool(k.value())
    return False


def _read_worker_mask(path: str) -> np.ndarray:
    """Read the worker's 16-bit grayscale mask PNG as float32 [0, 1]."""
    import cv2

    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Worker produced no readable mask at {path}")
    if raw.ndim > 2:
        raw = raw[..., 0]
    scale = 65535.0 if raw.dtype == np.uint16 else 255.0
    return raw.astype(np.float32) / scale


def _run_via_worker(node, extras: Dict[str, Any], label: str) -> None:
    """Render the frame, run one inference in the worker, apply the mask."""
    import nuke
    import tempfile
    from . import worker_client

    model = _get_model_request(node)
    params = _get_output_params(node)
    image_path = render_frame_to_png(node)
    mask_path = os.path.join(
        tempfile.gettempdir(),
        f"_h2samvit_workermask_{os.getpid()}.png",
    )

    request = {
        "cmd": "infer",
        "image": image_path,
        "out_mask": mask_path,
        "model": model,
        "params": params,
        "use_ma2": _use_ma2(node),
    }
    request.update(extras)

    try:
        resp = worker_client.request(request, label=label)
        mask = _read_worker_mask(mask_path)
    finally:
        for p in (image_path, mask_path):
            try:
                os.remove(p)
            except OSError:
                pass

    if params.get("temporal_on", False):
        from . import temporal
        mask = temporal.apply_consistency(node, mask, nuke.frame(), params)

    write_mask_to_node(node, mask, params)

    try:
        from . import model_manager
        node.knob("model_status").setValue(
            model_manager.status_text(model["family"], model["version"], model["size"])
        )
    except Exception:
        pass

    print(f"[H2 SamViT] {resp.get('message', 'Inference complete.')}")


def run_point_bbox_inference(
    node,
    points: List[Dict[str, Any]],
    bbox: Optional[Tuple[float, float, float, float]],
    neg_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> str:
    """Run SAM inference with points and/or bounding box prompts.

    Coordinates are passed in Nuke space (bottom-left origin); the worker
    flips them to SAM space using the rendered image height.

    Returns "ok", "cancelled" or "failed" — process_sequence uses this to
    stop the whole run when the user cancels one frame.
    """
    import nuke
    from . import worker_client

    try:
        _run_via_worker(
            node,
            {
                "mode": "point_bbox",
                "points": points,
                "bbox": list(bbox) if bbox else None,
                "neg_bbox": list(neg_bbox) if neg_bbox else None,
            },
            label="H2 SamViT — inference",
        )
        return "ok"
    except worker_client.WorkerCancelled:
        print("[H2 SamViT] Inference cancelled.")
        return "cancelled"
    except Exception as e:
        import traceback
        nuke.message(f"Inference failed: {str(e)}\n\n{traceback.format_exc()}")
        return "failed"


def run_text_prompt_inference(
    node,
    text_prompt: str,
    selection_points: List[Dict[str, Any]]
) -> str:
    """Run text-based detection + segmentation (SAM3 native / SAM2+DINO).

    Returns "ok", "cancelled" or "failed" (see run_point_bbox_inference).
    """
    import nuke
    from . import worker_client

    try:
        _run_via_worker(
            node,
            {
                "mode": "text",
                "text": text_prompt,
                "selection_points": selection_points,
            },
            label="H2 SamViT — text inference",
        )
        return "ok"
    except worker_client.WorkerCancelled:
        print("[H2 SamViT] Inference cancelled.")
        return "cancelled"
    except Exception as e:
        import traceback
        nuke.message(f"Text inference failed: {str(e)}\n\n{traceback.format_exc()}")
        return "failed"


def select_detection_by_point(
    boxes: np.ndarray,
    point: Dict[str, Any]
) -> int:
    """Select the detection box that contains or is closest to the point."""
    px, py = point["x"], point["y"]

    # First, check if point is inside any box
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if x1 <= px <= x2 and y1 <= py <= y2:
            return i

    # If not inside any box, find closest box center
    centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes]
    distances = [np.sqrt((c[0] - px) ** 2 + (c[1] - py) ** 2) for c in centers]

    return int(np.argmin(distances))


def _get_output_params(node) -> Dict[str, Any]:
    """Read all output / post-processing knobs from the gizmo."""
    def _val(name, default=0):
        k = node.knob(name)
        return k.value() if k else default

    return {
        "black_point": float(_val("black_point", 0.0)),
        "white_point": float(_val("white_point", 1.0)),
        "fill_holes": bool(_val("fill_holes", True)),
        "fill_holes_area": int(_val("fill_holes_area", 256)),
        "mask_shrink_grow": int(_val("mask_shrink_grow", 0)),
        "edge_feather": int(_val("edge_feather", 0)),
        "offset_x": int(_val("offset_x", 0)),
        "offset_y": int(_val("offset_y", 0)),
        "final_binary": bool(_val("final_binary", False)),
        "output_mode": str(_val("output_mode", "Straight")),
        "temporal_on": bool(_val("temporal_on", False)),
        "temporal_weight": float(_val("temporal_weight", 0.5)),
        "suppress_thresh": float(_val("suppress_thresh", 0.3)),
        "debug_save_coarse": bool(_val("debug_save_coarse", False)),
    }


def _debug_save(node, mask: np.ndarray, label: str):
    """Save a mask to temp dir for debugging."""
    import tempfile
    import cv2
    import nuke
    path = os.path.join(
        tempfile.gettempdir(),
        f"h2_samvit_{label}_{node.name()}_{nuke.frame()}.png",
    )
    cv2.imwrite(path, (np.clip(mask, 0, 1) * 255).astype(np.uint8))
    print(f"[H2 SamViT] Debug saved -> {path}")


def preprocess_mask(mask: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply pre-processing to the coarse mask."""
    import cv2

    mask = mask.copy()

    # Binarize at 0.5
    mask = (mask > 0.5).astype(np.float32)

    # Fill holes
    if params.get("fill_holes", True):
        hole_area = int(params.get("fill_holes_area", 256))
        if hole_area > 0:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                255 - mask_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for c in contours:
                if cv2.contourArea(c) < hole_area:
                    cv2.fillPoly(mask_uint8, [c], 255)
            mask = mask_uint8.astype(np.float32) / 255.0

    # Black point / white point levels
    bp = float(params.get("black_point", 0.0))
    wp = float(params.get("white_point", 1.0))
    if bp > 0.0 or wp < 1.0:
        wp = max(wp, bp + 0.001)
        mask = np.clip((mask - bp) / (wp - bp), 0.0, 1.0)

    return mask


def postprocess_mask(mask: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply post-processing to the refined mask."""
    import cv2

    mask = mask.copy()

    # Shrink/Grow
    shrink_grow = int(params.get("mask_shrink_grow", 0))
    if shrink_grow != 0:
        ksize = abs(shrink_grow) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask_uint8 = (mask * 255).astype(np.uint8)
        if shrink_grow > 0:
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
        else:
            mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)
        mask = mask_uint8.astype(np.float32) / 255.0

    # Edge feather
    feather = int(params.get("edge_feather", 0))
    if feather > 0:
        ksize = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    # Offset
    ox = int(params.get("offset_x", 0))
    oy = int(params.get("offset_y", 0))
    if ox != 0 or oy != 0:
        M = np.float32([[1, 0, ox], [0, 1, oy]])
        mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

    # Final binary
    if params.get("final_binary", False):
        mask = (mask > 0.5).astype(np.float32)

    return np.clip(mask, 0, 1)


def write_mask_to_node(node, mask: np.ndarray, params: Dict[str, Any]) -> None:
    """Write the computed mask into the gizmo's internal node graph.

    1. Saves the mask as a grayscale PNG in a temp directory.
    2. Inside the gizmo group, swaps the default Constant for a Read
       node (first run) or updates the existing Read node.
    3. Enables the CopyAlpha node so the mask feeds the output alpha.
    """
    import nuke
    import tempfile

    frame = nuke.frame()
    node_name = node.name()

    # ── Write mask file ──
    import cv2
    # Masks live NEXT TO THE SAVED SCRIPT so they survive temp cleanups and
    # travel with the comp (the pattern is baked into the saved .nk via the
    # internal Read node). Unsaved scripts fall back to %TEMP% — save before
    # inferring if the masks need to persist.
    script = nuke.root().name()
    if script and script != "Root":
        mask_root = os.path.join(os.path.dirname(script), "h2_samvit_masks")
    else:
        mask_root = os.path.join(tempfile.gettempdir(), "h2_samvit_masks")
    # Forward slashes throughout: this pattern lands in a Read node's file
    # knob, where Windows backslashes get mangled as escape sequences.
    mask_dir = os.path.join(mask_root, node_name).replace("\\", "/")
    os.makedirs(mask_dir, exist_ok=True)
    mask_path = f"{mask_dir}/mask.{frame:04d}.png"
    mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(mask_path, mask_uint8)

    mask_pattern = f"{mask_dir}/mask.####.png"

    # ── Update gizmo internals ──
    node.begin()
    try:
        mask_src = nuke.toNode("MaskSource")
        copy_alpha = nuke.toNode("CopyAlpha")
        input_node = nuke.toNode("Input1")

        if mask_src is not None and mask_src.Class() == "Constant":
            # First inference – swap Constant for a Read node
            xp, yp = mask_src.xpos(), mask_src.ypos()
            nuke.delete(mask_src)

            mask_read = nuke.nodes.Read()
            mask_read.setName("MaskSource")
            mask_read.setXpos(xp)
            mask_read.setYpos(yp)
            mask_read["file"].setValue(mask_pattern)
            mask_read["first"].setValue(frame)
            mask_read["last"].setValue(frame)
            mask_read["raw"].setValue(True)  # Mask is data — no colorspace transform

            # Re-wire the Copy node
            if copy_alpha and input_node:
                copy_alpha.setInput(0, input_node)
                copy_alpha.setInput(1, mask_read)

        elif mask_src is not None:
            # Subsequent inference – update the existing Read node
            mask_src["file"].setValue(mask_pattern)
            cur_first = int(mask_src["first"].value())
            cur_last = int(mask_src["last"].value())
            mask_src["first"].setValue(min(cur_first, frame))
            mask_src["last"].setValue(max(cur_last, frame))
            try:
                mask_src["reload"].execute()
            except Exception:
                pass

        # Enable the Copy node so the mask flows to the output
        if copy_alpha:
            copy_alpha["disable"].setValue(False)
    finally:
        node.end()

    # ── Also keep in-memory cache ──
    from . import cache
    cache.store_mask(node_name, frame, mask)

    nuke.updateUI()
    print(f"[H2 SamViT] Mask written – frame {frame}")


def clear_models():
    """Free the worker's models / GPU memory (and any legacy local state)."""
    global _sam_predictor, _sam3_model, _current_model_key, _text_model

    from . import worker_client
    worker_client.clear()

    with _model_lock:
        _sam_predictor = None
        _sam3_model = None
        _current_model_key = None
        _text_model = None

    print("[H2 SamViT] Models cleared from memory.")
