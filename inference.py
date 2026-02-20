# inference.py - SAM3 + ViTMatte inference engine for H2 SamViT
# Handles model loading, inference, and result processing

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


# Global model instances (lazy loaded)
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

    For **SAM2** — builds a ``SAM2ImagePredictor`` via local YAML config.
    For **SAM3** — builds the full ``Sam3Image`` model, then exposes
    its ``.inst_interactive_predictor`` (``SAM3InteractiveImagePredictor``)
    which shares the same ``set_image`` / ``predict`` API as
    ``SAM2ImagePredictor``.

    Returns the predictor (point/bbox compatible).
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


def image_from_nuke_node(node) -> np.ndarray:
    """Extract the current frame from the node's input as a uint8 RGB array.

    Always renders through a temporary Write node so that **every**
    format Nuke can decode is supported (EXR, TIFF, MOV, MP4, PNG,
    JPG, DPX, etc.) and any upstream colour-space / transform
    operations are baked in.
    """
    import nuke
    import tempfile
    import cv2

    input_node = node.input(0)
    if not input_node:
        raise ValueError("No input connected to H2_SamViT node.")

    frame = nuke.frame()

    # Render via temp PNG — works with every format Nuke can decode.
    tmp = os.path.join(
        tempfile.gettempdir(),
        f"_h2samvit_input_{os.getpid()}_{frame}.png",
    )

    write = nuke.nodes.Write()
    write["file"].setValue(tmp)
    write["file_type"].setValue("png")
    write.setInput(0, input_node)

    try:
        nuke.execute(write, frame, frame)

        bgr = cv2.imread(tmp, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(
                f"Could not read the rendered frame from {tmp}"
            )
        print(f"[H2 SamViT] Input captured — frame {frame}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        nuke.delete(write)
        try:
            os.remove(tmp)
        except OSError:
            pass


def refine_mask_with_vitmatte(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    erode_radius: int = 5,
    dilate_radius: int = 15,
    crop_padding: float = 20.0,
) -> np.ndarray:
    """Refine a coarse mask using ViTMatte for production-quality alpha.

    Uses cropped refinement — ViTMatte only processes the masked region
    plus *crop_padding* %, giving more detail resolution on the actual
    edges (hair, fur, translucency) where it matters most.
    """
    from . import vitmatte_refiner
    refiner = vitmatte_refiner.get_refiner()
    return refiner.refine_with_crop(
        image, coarse_mask,
        padding_percent=crop_padding,
        erode_radius=erode_radius,
        dilate_radius=dilate_radius,
    )


def run_point_bbox_inference(
    node,
    points: List[Dict[str, Any]],
    bbox: Optional[Tuple[float, float, float, float]],
    neg_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """Run SAM inference with points and/or bounding box prompts.

    Works identically for SAM2 and SAM3 because both
    ``SAM2ImagePredictor`` and ``SAM3InteractiveImagePredictor``
    share the same ``set_image`` / ``predict`` API.

    For SAM3 with a negative bbox: the neg bbox corners are added as
    background (label=0) points to steer the model away from that
    region — SAM's instance segmentation path does not natively support
    negative boxes, but background points at the neg bbox corners
    achieve the same effect.

    Following the official SAM2/SAM3 inference pattern:
      • ``torch.inference_mode()`` for maximum efficiency
      • ``torch.autocast`` for mixed-precision (bf16/fp16)
      • Model weights stay in float32 — autocast handles precision
      • ``multimask_output`` is True for single-point prompts (ambiguous)
        and False for multi-point or box prompts (clear intent)
    """
    import torch
    import nuke

    try:
        sam_predictor = load_sam_model(node)

        image = image_from_nuke_node(node)

        # Determine autocast context from the precision knob.
        # Both SAM2 and SAM3 use the same pattern: fp32 weights + autocast.
        precision = node.knob("model_precision").value()
        amp_ctx = _resolve_autocast_ctx(precision)

        # Image height needed to flip Y coordinates from Nuke
        # (bottom-left origin) to SAM (top-left origin).
        img_h = image.shape[0]

        with torch.inference_mode(), amp_ctx:
            sam_predictor.set_image(image)

            # Prepare prompts
            point_coords = None
            point_labels = None
            box = None

            if points:
                point_coords = np.array(
                    [[p["x"], img_h - p["y"]] for p in points],
                    dtype=np.float32,
                )
                point_labels = np.array(
                    [p["label"] for p in points],
                    dtype=np.int32,
                )

            if bbox:
                x1, y1_nk, x2, y2_nk = bbox
                box = np.array(
                    [x1, img_h - y2_nk, x2, img_h - y1_nk],
                    dtype=np.float32,
                )

            # Negative bbox → add its 4 corners as background points
            # (label=0).  SAM's instance segmentation path doesn't
            # natively support negative boxes, but background points at
            # the corners effectively exclude that region.
            if neg_bbox:
                nx1, ny1_nk, nx2, ny2_nk = neg_bbox
                neg_corners = np.array([
                    [nx1, img_h - ny2_nk],
                    [nx2, img_h - ny2_nk],
                    [nx1, img_h - ny1_nk],
                    [nx2, img_h - ny1_nk],
                    [(nx1 + nx2) / 2, (img_h - ny2_nk + img_h - ny1_nk) / 2],
                ], dtype=np.float32)
                neg_labels = np.zeros(len(neg_corners), dtype=np.int32)

                if point_coords is not None:
                    point_coords = np.concatenate(
                        [point_coords, neg_corners], axis=0)
                    point_labels = np.concatenate(
                        [point_labels, neg_labels], axis=0)
                else:
                    point_coords = neg_corners
                    point_labels = neg_labels

            # Smart multimask_output:
            # - 1 point, no box → True  (ambiguous prompt, pick best of 3)
            # - 2+ points or box → False (clear intent, single best mask)
            num_prompts = (len(points) if points else 0) + (1 if bbox else 0)
            use_multimask = num_prompts <= 1

            masks, scores, logits = sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=use_multimask,
            )

        best_idx = int(np.argmax(scores))
        coarse_mask = masks[best_idx]

        _refine_and_write(
            node, image, coarse_mask,
            f"Inference complete. Score: {scores[best_idx]:.3f}",
        )

    except Exception as e:
        import traceback
        nuke.message(f"Inference failed: {str(e)}\n\n{traceback.format_exc()}")


def run_text_prompt_inference(
    node,
    text_prompt: str,
    selection_points: List[Dict[str, Any]]
) -> None:
    """Run text-based object detection + segmentation.

    • **SAM3** — uses ``Sam3Processor`` (built-in text grounding via
      CLIP language backbone).  No external detector needed.
    • **SAM2** — falls back to Grounding DINO for detection,
      then feeds the detected box to ``SAM2ImagePredictor``.
    """
    import torch
    import nuke

    family = node.knob("model_family").value()

    try:
        if family == "SAM3":
            _run_text_inference_sam3(node, text_prompt, selection_points)
        else:
            _run_text_inference_sam2(node, text_prompt, selection_points)
    except Exception as e:
        import traceback
        nuke.message(f"Text inference failed: {str(e)}\n\n{traceback.format_exc()}")


def _run_text_inference_sam3(
    node,
    text_prompt: str,
    selection_points: List[Dict[str, Any]],
) -> None:
    """SAM3 text-prompt inference using its built-in Sam3Processor."""
    import torch
    import nuke
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor

    # Ensure model is loaded (also populates _sam3_model)
    load_sam_model(node)

    if _sam3_model is None:
        raise RuntimeError("SAM3 model is not loaded.")

    image = image_from_nuke_node(node)                   # uint8 RGB np array
    pil_image = Image.fromarray(image)

    processor = Sam3Processor(_sam3_model)

    # Use precision from the knob — autocast handles mixed precision.
    precision = node.knob("model_precision").value()
    amp_ctx = _resolve_autocast_ctx(precision)

    with torch.inference_mode(), amp_ctx:
        state  = processor.set_image(pil_image)
        output = processor.set_text_prompt(text_prompt, state)

    masks  = output["masks"]
    scores = output["scores"]
    boxes  = output.get("boxes", None)

    if hasattr(masks, "cpu"):
        masks = masks.cpu().float().numpy()
    if hasattr(scores, "cpu"):
        scores = scores.cpu().float().numpy()
    if boxes is not None and hasattr(boxes, "cpu"):
        boxes = boxes.cpu().float().numpy()

    if len(masks) == 0:
        nuke.message(f"No objects matching '{text_prompt}' found.")
        return

    # Select which detection to use
    if selection_points and boxes is not None and len(boxes) > 0:
        selected_idx = select_detection_by_point(boxes, selection_points[0])
    else:
        selected_idx = int(np.argmax(scores))

    coarse_mask = masks[selected_idx].astype(np.float32)

    while coarse_mask.ndim > 2:
        coarse_mask = coarse_mask.squeeze(0)

    # ── If the text detection found a bounding box, refine using that
    # box as an additional SAM point/bbox prompt for better coverage ──
    if boxes is not None and len(boxes) > 0 and _sam3_model is not None:
        try:
            sel_box = boxes[selected_idx]
            predictor = _sam3_model.inst_interactive_predictor
            sam_box = np.array([
                sel_box[0], sel_box[1], sel_box[2], sel_box[3]
            ])

            with torch.inference_mode(), amp_ctx:
                predictor.set_image(image)
                refined_masks, refined_scores, _ = predictor.predict(
                    box=sam_box,
                    multimask_output=False,
                )

            best_refined = int(np.argmax(refined_scores))
            refined_mask = refined_masks[best_refined].astype(np.float32)
            while refined_mask.ndim > 2:
                refined_mask = refined_mask.squeeze(0)

            # Use refined mask if it has better coverage
            if refined_mask.sum() > coarse_mask.sum() * 0.5:
                coarse_mask = np.maximum(coarse_mask, refined_mask)
                print("[H2 SamViT] Text mask refined with box prompt")
        except Exception as e:
            print(f"[H2 SamViT] Box refinement skipped: {e}")

    # ── Shared refinement pipeline ──
    _refine_and_write(node, image, coarse_mask,
                      f"Text inference (SAM3) complete. "
                      f"Found {len(masks)} objects, selected #{selected_idx + 1}")


def _run_text_inference_sam2(
    node,
    text_prompt: str,
    selection_points: List[Dict[str, Any]],
) -> None:
    """SAM2 text-prompt inference: Grounding DINO detection → SAM2 segmentation."""
    import torch
    import nuke

    text_detector = load_text_model()
    sam_predictor = load_sam_model(node)

    image = image_from_nuke_node(node)

    # ── Grounding DINO detection ──
    model     = text_detector["model"]
    processor = text_detector["processor"]
    device    = text_detector["device"]

    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=0.2,
        text_threshold=0.2,
        target_sizes=[(image.shape[0], image.shape[1])]
    )[0]

    boxes  = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    if len(boxes) == 0:
        nuke.message(f"No objects matching '{text_prompt}' found.")
        return

    if selection_points:
        selected_idx = select_detection_by_point(boxes, selection_points[0])
    else:
        selected_idx = int(np.argmax(scores))

    selected_box = boxes[selected_idx]

    # ── SAM2 segmentation with detected box ──
    precision = node.knob("model_precision").value()
    amp_ctx = _resolve_autocast_ctx(precision)

    with torch.inference_mode(), amp_ctx:
        sam_predictor.set_image(image)

        masks, mask_scores, logits = sam_predictor.predict(
            box=selected_box,
            multimask_output=False,
        )

    best_idx = int(np.argmax(mask_scores))
    coarse_mask = masks[best_idx]

    _refine_and_write(node, image, coarse_mask,
                      f"Text inference (SAM2) complete. "
                      f"Found {len(boxes)} objects, selected #{selected_idx + 1}")


def _refine_and_write(
    node,
    image: np.ndarray,
    coarse_mask: np.ndarray,
    log_message: str,
) -> None:
    """Shared refinement / post-processing / write-back pipeline.

    Alpha handling follows the same pattern as ComfyUI-SAM2 / SAM3:
    SAM's ``predict()`` returns boolean masks — ``masks > threshold``
    — which are pure binary (0 or 1).  We keep them binary by default
    so the resulting alpha channel is **always** pure white (1.0) or
    pure black (0.0) and is never affected by upstream colour grading.

    ViTMatte soft-alpha refinement is available behind the
    ``use_vitmatte`` toggle for compositing workflows that need it.
    """
    import nuke
    from . import callbacks

    # Guarantee the mask is 2-D (H, W).
    while coarse_mask.ndim > 2:
        coarse_mask = coarse_mask.squeeze(0)

    params = callbacks.get_inference_params(node)

    coarse_mask = preprocess_mask(coarse_mask, params)

    # ── ViTMatte refinement (optional — OFF by default) ──
    use_vitmatte = params.get("use_vitmatte", False)

    if use_vitmatte:
        erode_radius  = int(params.get("trimap_erode_radius", 5))
        dilate_radius = int(params.get("trimap_dilate_radius", 15))
        crop_padding  = float(params.get("crop_padding", 20.0))

        alpha_matte = refine_mask_with_vitmatte(
            image, coarse_mask,
            erode_radius=erode_radius,
            dilate_radius=dilate_radius,
            crop_padding=crop_padding,
        )

        # Normalize: ensure the FG core reaches 1.0
        alpha_max = float(alpha_matte.max())
        if 0.01 < alpha_max < 0.95:
            alpha_matte = np.clip(alpha_matte / alpha_max, 0.0, 1.0)
        else:
            alpha_matte = np.clip(alpha_matte, 0.0, 1.0)

        # Clean near-zero noise in definite BG areas
        alpha_matte[alpha_matte < 0.004] = 0.0

        # CRITICAL: ViTMatte produces proper soft alpha with natural
        # edge transitions — final_binary_sharp MUST be disabled or
        # it converts the soft matte back to hard 0/1 and destroys
        # all the edge quality ViTMatte just computed.
        params = dict(params)  # local copy — don't mutate original
        params["final_binary_sharp"] = False

        print(f"[H2 SamViT] ViTMatte refinement applied "
              f"(inner={erode_radius}px, outer={dilate_radius}px)")

        # Debug: save trimap to disk if requested
        if params.get("show_trimap_overlay", False):
            try:
                import tempfile
                from . import vitmatte_refiner
                refiner = vitmatte_refiner.get_refiner()
                trimap = refiner.create_trimap(
                    coarse_mask,
                    erode_radius=erode_radius,
                    dilate_radius=dilate_radius,
                )
                trimap_path = os.path.join(
                    tempfile.gettempdir(),
                    f"h2_samvit_trimap_{node.name()}.png",
                )
                import cv2
                cv2.imwrite(trimap_path, trimap)
                print(f"[H2 SamViT] Trimap debug saved \u2192 {trimap_path}")
            except Exception:
                pass
    else:
        # Pure binary mask — matches ComfyUI-SAM2/SAM3 behaviour.
        # Alpha is strictly 0.0 or 1.0 and cannot be affected by grading.
        alpha_matte = (coarse_mask > 0.5).astype(np.float32)

    alpha_matte = postprocess_mask(alpha_matte, params)

    if params["enable_temporal_consistency"]:
        from . import temporal
        frame = nuke.frame()
        alpha_matte = temporal.apply_consistency(node, alpha_matte, frame, params)

    write_mask_to_node(node, alpha_matte, params)

    # ── Render the coloured overlay (points + bbox) ──
    # Safe here because we are called from a button callback,
    # not from knobChanged.
    try:
        from . import ui_overlay
        ui_overlay.render_overlay(node)
    except Exception:
        pass

    print(f"[H2 SamViT] {log_message}")


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


def preprocess_mask(mask: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply pre-processing to the coarse mask."""
    import cv2
    
    mask = mask.copy()
    
    # Binary threshold
    threshold = params["input_threshold"] / 255.0
    mask = (mask > threshold).astype(np.float32)
    
    # Fill holes
    if params.get("fill_holes", True):
        hole_fill_area = int(params.get("fill_holes_area", 16))
        if hole_fill_area > 0:
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Find contours and fill small holes
            contours, _ = cv2.findContours(
                255 - mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < hole_fill_area ** 2:
                    cv2.fillPoly(mask_uint8, [contour], 255)
            
            mask = mask_uint8.astype(np.float32) / 255.0
    
    # Black point / white point levels
    bp = float(params.get("black_point", 0.0))
    wp = float(params.get("white_point", 1.0))
    if bp > 0.0 or wp < 1.0:
        wp = max(wp, bp + 0.001)  # prevent division by zero
        mask = np.clip((mask - bp) / (wp - bp), 0.0, 1.0)
    
    return mask


def postprocess_mask(mask: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply post-processing to the refined mask."""
    import cv2
    
    mask = mask.copy()
    
    # Shrink/Grow
    shrink_grow = int(params["mask_shrink_grow"])
    if shrink_grow != 0:
        mask_uint8 = (mask * 255).astype(np.uint8)
        kernel = np.ones((abs(shrink_grow), abs(shrink_grow)), np.uint8)
        
        if shrink_grow > 0:
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
        else:
            mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)
        
        mask = mask_uint8.astype(np.float32) / 255.0
    
    # Edge feather
    feather = int(params["edge_feather"])
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)
    
    # Offset
    offset_x = int(params["offset_mask_x"])
    offset_y = int(params["offset_mask_y"])
    if offset_x != 0 or offset_y != 0:
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    
    # Final binary sharp — ensures pure 0/1 alpha
    if params["final_binary_sharp"]:
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
    mask_dir = os.path.join(
        tempfile.gettempdir(), "h2_samvit_masks", node_name
    )
    os.makedirs(mask_dir, exist_ok=True)
    mask_path = os.path.join(mask_dir, f"mask.{frame:04d}.png")
    mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(mask_path, mask_uint8)

    mask_pattern = os.path.join(mask_dir, "mask.####.png")

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
    """Clear loaded models to free GPU memory."""
    global _text_model

    with _model_lock:
        _free_models()
        _text_model = None

    # Unload ViTMatte via refiner
    try:
        from . import vitmatte_refiner
        refiner = vitmatte_refiner.get_refiner()
        if refiner._loaded:
            refiner.unload()
    except Exception:
        pass

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[H2 SamViT] Models cleared from memory.")
