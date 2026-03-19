# callbacks_v3.py — Nuke callback handlers for H2 SamViT v3
#
# Reads point/bbox data from XY knobs on the gizmo panel.
# Drives SAM3/SAM2 + MatAnyone2 inference pipeline.
# Supports non-blocking inference via threading.

import nuke
import os
import threading
from typing import List, Dict, Any, Optional, Tuple

MAX_FG = 4
MAX_BG = 4

_in_callback = False


# ─────────────────────────────────────────────────────────────────
#  Point / Bbox reading
# ─────────────────────────────────────────────────────────────────

def _xy(node, knob_name) -> Tuple[float, float]:
    """Read an XY_Knob as (x, y)."""
    k = node.knob(knob_name)
    if k is None:
        return (0.0, 0.0)
    return (k.value(0), k.value(1))


def get_fg_points(node) -> List[Dict[str, Any]]:
    """Return enabled foreground points [{x, y, label:1}, ...]."""
    pts = []
    for i in range(1, MAX_FG + 1):
        if node.knob(f"fg_{i}_on") and node.knob(f"fg_{i}_on").value():
            x, y = _xy(node, f"fg_{i}")
            pts.append({"x": x, "y": y, "label": 1})
    return pts


def get_bg_points(node) -> List[Dict[str, Any]]:
    """Return enabled background points [{x, y, label:0}, ...]."""
    pts = []
    for i in range(1, MAX_BG + 1):
        if node.knob(f"bg_{i}_on") and node.knob(f"bg_{i}_on").value():
            x, y = _xy(node, f"bg_{i}")
            pts.append({"x": x, "y": y, "label": 0})
    return pts


def get_bbox(node) -> Optional[Tuple[float, float, float, float]]:
    """Return (x_min, y_min, x_max, y_max) or None if disabled."""
    if not (node.knob("bbox_on") and node.knob("bbox_on").value()):
        return None
    tl = _xy(node, "bbox_tl")
    br = _xy(node, "bbox_br")
    x1, y1 = min(tl[0], br[0]), min(tl[1], br[1])
    x2, y2 = max(tl[0], br[0]), max(tl[1], br[1])
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)


def get_neg_bbox(node) -> Optional[Tuple[float, float, float, float]]:
    """Return negative bbox or None."""
    if not (node.knob("neg_bbox_on") and node.knob("neg_bbox_on").value()):
        return None
    tl = _xy(node, "neg_bbox_tl")
    br = _xy(node, "neg_bbox_br")
    x1, y1 = min(tl[0], br[0]), min(tl[1], br[1])
    x2, y2 = max(tl[0], br[0]), max(tl[1], br[1])
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)


def collect_prompt_data(node) -> Dict[str, Any]:
    """Gather all prompt data for inference."""
    mode = node.knob("pipeline_mode").value()
    data = {
        "pipeline_mode": mode,
        "points": get_fg_points(node) + get_bg_points(node),
        "pos_bbox": get_bbox(node),
        "neg_bbox": get_neg_bbox(node),
        "text_prompt": None,
    }
    if mode == "Text Prompt":
        k = node.knob("text_prompt")
        data["text_prompt"] = k.value() if k else ""
    return data


# ─────────────────────────────────────────────────────────────────
#  Point manipulation
# ─────────────────────────────────────────────────────────────────

def add_fg_point(node):
    """Enable the next available FG point slot."""
    for i in range(1, MAX_FG + 1):
        k = node.knob(f"fg_{i}_on")
        if k and not k.value():
            k.setValue(True)
            # Centre on the input format
            inp = node.input(0)
            if inp:
                fmt = inp.format()
                node.knob(f"fg_{i}").setValue([fmt.width() / 2, fmt.height() / 2])
            nuke.message(f"FG {i} enabled.\nSet its coordinates in the properties panel.")
            return
    nuke.message(f"All {MAX_FG} FG point slots are in use.")


def add_bg_point(node):
    """Enable the next available BG point slot."""
    for i in range(1, MAX_BG + 1):
        k = node.knob(f"bg_{i}_on")
        if k and not k.value():
            k.setValue(True)
            inp = node.input(0)
            if inp:
                fmt = inp.format()
                node.knob(f"bg_{i}").setValue([fmt.width() / 2 + 100, fmt.height() / 2 + 100])
            nuke.message(f"BG {i} enabled.\nSet its coordinates in the properties panel.")
            return
    nuke.message(f"All {MAX_BG} BG point slots are in use.")


def clear_fg_points(node):
    """Disable all FG point slots."""
    for i in range(1, MAX_FG + 1):
        k = node.knob(f"fg_{i}_on")
        if k:
            k.setValue(False)


def clear_bg_points(node):
    """Disable all BG point slots."""
    for i in range(1, MAX_BG + 1):
        k = node.knob(f"bg_{i}_on")
        if k:
            k.setValue(False)


def reset_bbox(node):
    """Reset bbox knobs to default centred rectangle."""
    k = node.knob("bbox_on")
    if k:
        k.setValue(False)
    inp = node.input(0)
    if inp:
        fmt = inp.format()
        w, h = fmt.width(), fmt.height()
    else:
        w, h = 1920, 1080
    node.knob("bbox_tl").setValue([w * 0.25, h * 0.25])
    node.knob("bbox_br").setValue([w * 0.75, h * 0.75])

    k2 = node.knob("neg_bbox_on")
    if k2:
        k2.setValue(False)
    node.knob("neg_bbox_tl").setValue([100, 100])
    node.knob("neg_bbox_br").setValue([300, 300])


# ─────────────────────────────────────────────────────────────────
#  onCreate / knobChanged
# ─────────────────────────────────────────────────────────────────

def on_create(node):
    """Called when gizmo is created."""
    print(f"[H2 SamViT] Node created: {node.fullName()}")
    _update_status(node, " ")


def on_knob_changed(node, knob):
    """Handle knob changes — show/hide conditional UI elements."""
    global _in_callback
    if _in_callback:
        return
    _in_callback = True

    try:
        name = knob.name()

        if name == "pipeline_mode":
            mode = knob.value()
            tp = node.knob("text_prompt")
            if tp:
                tp.setVisible(mode == "Text Prompt")

        elif name == "model_family":
            family = knob.value()
            # SAM version / size only apply to SAM2
            for kn in ("sam_version", "model_size"):
                k = node.knob(kn)
                if k:
                    k.setVisible(family == "SAM2")
            nk = node.knob("neg_bbox_on")
            if nk and family != "SAM3":
                nk.setValue(False)

        elif name == "cache_pct":
            _update_cache_info(node)
    finally:
        _in_callback = False


# ─────────────────────────────────────────────────────────────────
#  Inference execution
# ─────────────────────────────────────────────────────────────────

def _update_status(node, text: str):
    """Update the inference status label."""
    try:
        k = node.knob("inference_status")
        if k:
            k.setValue(text)
    except Exception:
        pass


def run_inference(node):
    """Run SAM + MatAnyone2 inference on the current frame.

    Runs in the main thread (Nuke API calls must happen on main thread).
    Uses torch.inference_mode + autocast for efficiency.
    """
    _update_status(node, "Running inference...")
    prompt = collect_prompt_data(node)

    n_pts = len(prompt["points"])
    has_box = prompt["pos_bbox"] is not None
    has_text = bool(prompt.get("text_prompt"))
    mode = prompt["pipeline_mode"]

    if mode == "Text Prompt" and not has_text:
        nuke.message("Enter a text prompt first.")
        _update_status(node, "No text prompt")
        return

    if mode == "Point / Bbox" and n_pts == 0 and not has_box:
        nuke.message("Add at least one point or enable the bounding box.")
        _update_status(node, "No prompts")
        return

    try:
        from . import inference

        if mode == "Text Prompt":
            inference.run_text_prompt_inference(
                node,
                prompt["text_prompt"],
                prompt["points"],
            )
        else:
            inference.run_point_bbox_inference(
                node,
                prompt["points"],
                prompt["pos_bbox"],
                prompt["neg_bbox"],
            )
        _update_status(node, f"Done  (frame {nuke.frame()})")
    except Exception as e:
        import traceback
        traceback.print_exc()
        _update_status(node, f"Error: {e}")
        nuke.message(f"Inference error:\n{e}")


def process_sequence(node):
    """Process the full frame range using SAM + MatAnyone2 temporal matting.

    For sequences, SAM produces the coarse mask on the first frame,
    then MatAnyone2 propagates the alpha matte across all frames
    using its temporal memory.
    """
    first = int(nuke.root().firstFrame())
    last = int(nuke.root().lastFrame())
    total = last - first + 1

    if total < 1:
        nuke.message("No valid frame range.")
        return

    prompt = collect_prompt_data(node)
    if len(prompt["points"]) == 0 and prompt["pos_bbox"] is None:
        nuke.message("Add at least one point or bbox for the first frame.")
        return

    _update_status(node, f"Processing {total} frames...")

    try:
        from . import inference

        # 1. Get coarse mask on first frame
        nuke.frame(first)
        image_first = inference.image_from_nuke_node(node)
        sam_pred = inference.load_sam_model(node)

        import numpy as np
        import torch

        img_h = image_first.shape[0]
        precision = node.knob("model_precision").value()
        amp_ctx = inference._resolve_autocast_ctx(precision)

        with torch.inference_mode(), amp_ctx:
            sam_pred.set_image(image_first)

            points = prompt["points"]
            point_coords = None
            point_labels = None
            box = None

            if points:
                point_coords = np.array(
                    [[p["x"], img_h - p["y"]] for p in points], dtype=np.float32
                )
                point_labels = np.array([p["label"] for p in points], dtype=np.int32)

            bbox = prompt["pos_bbox"]
            if bbox:
                x1, y1, x2, y2 = bbox
                box = np.array([x1, img_h - y2, x2, img_h - y1], dtype=np.float32)

            n_prompts = (len(points) if points else 0) + (1 if bbox else 0)
            masks, scores, _ = sam_pred.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=n_prompts <= 1,
            )

        best = int(np.argmax(scores))
        coarse_mask = masks[best].astype(np.float32)
        while coarse_mask.ndim > 2:
            coarse_mask = coarse_mask.squeeze(0)

        # 2. Collect all frames as images
        images = [image_first]
        for f in range(first + 1, last + 1):
            nuke.frame(f)
            images.append(inference.image_from_nuke_node(node))

        # 3. Refine with MatAnyone2 sequence matting
        use_ma2 = node.knob("use_matanyone2").value()
        params = inference._get_output_params(node)

        if use_ma2:
            from . import matanyone2_refiner

            mem_every = int(node.knob("ma2_mem_every").value())
            max_mem = int(node.knob("ma2_max_mem").value())

            def _progress(idx, tot):
                _update_status(node, f"MatAnyone2: frame {idx + 1}/{tot}")

            alphas = matanyone2_refiner.refine_sequence(
                images, coarse_mask,
                mem_every=mem_every,
                max_mem_frames=max_mem,
                progress_callback=_progress,
            )
        else:
            alphas = [(coarse_mask > 0.5).astype(np.float32)] * len(images)

        # 4. Post-process and write each frame
        for i, alpha in enumerate(alphas):
            frame = first + i
            nuke.frame(frame)
            alpha = inference.postprocess_mask(alpha, params)
            inference.write_mask_to_node(node, alpha, params)

        _update_status(node, f"Sequence done ({total} frames)")
        nuke.updateUI()

    except Exception as e:
        import traceback
        traceback.print_exc()
        _update_status(node, f"Error: {e}")
        nuke.message(f"Sequence error:\n{e}")


# ─────────────────────────────────────────────────────────────────
#  Utility buttons
# ─────────────────────────────────────────────────────────────────

def clear_mask(node):
    """Reset the mask output to black (no alpha)."""
    node.begin()
    try:
        ms = nuke.toNode("MaskSource")
        ca = nuke.toNode("CopyAlpha")
        if ca:
            ca["disable"].setValue(True)
        if ms and ms.Class() != "Constant":
            xp, yp = ms.xpos(), ms.ypos()
            nuke.delete(ms)
            c = nuke.nodes.Constant()
            c.setName("MaskSource")
            c.setXpos(xp)
            c.setYpos(yp)
            c["channels"].setValue("alpha")
            c["color"].setValue(0)
    finally:
        node.end()
    nuke.updateUI()
    _update_status(node, "Mask cleared")


def clear_cache(node):
    """Clear in-memory mask cache."""
    from . import cache
    cache.clear_all_cache()
    _update_cache_info(node)


def unload_models(node):
    """Free all model memory."""
    from . import inference
    inference.clear_models()
    node.knob("model_status").setValue("Not loaded")
    node.knob("ma2_status").setValue("Not loaded")
    _update_status(node, "Models unloaded")


def download_model(node):
    """Download the selected SAM model checkpoint."""
    from . import model_manager
    family = node.knob("model_family").value()
    version = node.knob("sam_version").value() if family == "SAM2" else "3.0"
    size = node.knob("model_size").value() if family == "SAM2" else "Default"

    info = model_manager.get_info(family, version, size)
    if model_manager.is_downloaded(family, version, size):
        nuke.message(f"{family} checkpoint already present.")
        node.knob("model_status").setValue(model_manager.status_text(family, version, size))
        return

    label = f"{family} v{version} {size}" if family == "SAM2" else "SAM3"
    if not nuke.ask(f"Download {label}?  (~{info['mb']} MB)"):
        return

    try:
        model_manager.download(family, version, size)
        node.knob("model_status").setValue(model_manager.status_text(family, version, size))
    except Exception as e:
        nuke.message(f"Download failed:\n{e}")


def download_matanyone2(node):
    """Download the MatAnyone2 checkpoint."""
    from . import model_manager
    if model_manager.is_downloaded("MatAnyone2"):
        nuke.message("MatAnyone2 checkpoint already present.")
        node.knob("ma2_status").setValue("Ready")
        return

    info = model_manager.get_info("MatAnyone2")
    if not nuke.ask(f"Download MatAnyone2?  (~{info['mb']} MB)"):
        return

    try:
        model_manager.download("MatAnyone2")
        node.knob("ma2_status").setValue("Ready")
    except Exception as e:
        nuke.message(f"Download failed:\n{e}")


def _update_cache_info(node):
    """Update cache label from current cache stats."""
    try:
        from . import cache
        info = cache.get_cache().get_cache_info()
        k = node.knob("cache_info")
        if k:
            k.setValue(info)
    except Exception:
        pass
