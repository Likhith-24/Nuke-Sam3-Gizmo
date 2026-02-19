# inference.py - SAM3 + ViTMatte inference engine for H2 SamViT
# Handles model loading, inference, and result processing

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import threading

# Global model instances (lazy loaded)
_sam_model = None
_sam_predictor = None
_vitmatte_model = None
_text_model = None
_model_lock = threading.Lock()


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sam3_model():
    """Load SAM3 (Segment Anything Model 3) model."""
    global _sam_model, _sam_predictor
    
    if _sam_model is not None:
        return _sam_predictor
    
    with _model_lock:
        if _sam_model is not None:
            return _sam_predictor
        
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        device = get_device()
        print(f"[H2 SamViT] Loading SAM3 model on {device}...")
        
        # SAM3 model checkpoint path
        # Users should download from Meta's official release
        model_cfg = "sam2_hiera_l.yaml"
        checkpoint = os.path.join(
            os.path.dirname(__file__), 
            "models", 
            "sam2_hiera_large.pt"
        )
        
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"SAM3 checkpoint not found at {checkpoint}. "
                "Please download from https://github.com/facebookresearch/segment-anything-2"
            )
        
        _sam_model = build_sam2(model_cfg, checkpoint, device=device)
        _sam_predictor = SAM2ImagePredictor(_sam_model)
        
        print("[H2 SamViT] SAM3 model loaded successfully.")
        return _sam_predictor


def load_vitmatte_model():
    """Load ViTMatte model from HuggingFace."""
    global _vitmatte_model
    
    if _vitmatte_model is not None:
        return _vitmatte_model
    
    with _model_lock:
        if _vitmatte_model is not None:
            return _vitmatte_model
        
        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        
        device = get_device()
        print(f"[H2 SamViT] Loading ViTMatte model on {device}...")
        
        model_name = "hustvl/vitmatte-small-composition-1k"
        
        _vitmatte_model = {
            "model": VitMatteForImageMatting.from_pretrained(model_name).to(device),
            "processor": VitMatteImageProcessor.from_pretrained(model_name),
            "device": device
        }
        
        _vitmatte_model["model"].eval()
        
        print("[H2 SamViT] ViTMatte model loaded successfully.")
        return _vitmatte_model


def load_text_model():
    """Load text-to-detection model (Grounding DINO style)."""
    global _text_model
    
    if _text_model is not None:
        return _text_model
    
    with _model_lock:
        if _text_model is not None:
            return _text_model
        
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        
        device = get_device()
        print(f"[H2 SamViT] Loading text detection model on {device}...")
        
        model_name = "IDEA-Research/grounding-dino-base"
        
        _text_model = {
            "model": AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device),
            "processor": AutoProcessor.from_pretrained(model_name),
            "device": device
        }
        
        _text_model["model"].eval()
        
        print("[H2 SamViT] Text detection model loaded successfully.")
        return _text_model


def image_from_nuke_node(node) -> np.ndarray:
    """Extract image data from a Nuke node's input."""
    import nuke
    
    # Get input node
    input_node = node.input(0)
    if not input_node:
        raise ValueError("No input connected to the node.")
    
    # Get format
    fmt = input_node.format()
    width = fmt.width()
    height = fmt.height()
    
    # Sample the image
    # Create a temporary tile to fetch pixel data
    # In real implementation, this would use Nuke's C++ API for efficiency
    channels = ["rgba.red", "rgba.green", "rgba.blue"]
    
    # Placeholder - actual implementation needs Nuke's pixel access
    # This would typically use nukescripts or a C++ plugin
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    return image


def create_trimap(mask: np.ndarray, width: int = 10, erode_radius: int = 2, dilate_radius: int = 2) -> np.ndarray:
    """Create a trimap from a binary mask."""
    import cv2
    
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Apply erode/dilate cleanup
    if erode_radius > 0:
        erode_kernel = np.ones((erode_radius * 2 + 1, erode_radius * 2 + 1), np.uint8)
        mask_uint8 = cv2.erode(mask_uint8, erode_kernel, iterations=1)
    
    if dilate_radius > 0:
        dilate_kernel = np.ones((dilate_radius * 2 + 1, dilate_radius * 2 + 1), np.uint8)
        mask_uint8 = cv2.dilate(mask_uint8, dilate_kernel, iterations=1)
    
    # Erode for definite foreground
    kernel = np.ones((width, width), np.uint8)
    fg = cv2.erode(mask_uint8, kernel, iterations=1)
    
    # Dilate for definite background (inverted)
    bg = cv2.dilate(mask_uint8, kernel, iterations=1)
    
    # Create trimap: 0 = bg, 128 = unknown, 255 = fg
    trimap = np.zeros_like(mask_uint8)
    trimap[bg == 255] = 128  # Unknown region
    trimap[fg == 255] = 255  # Definite foreground
    # Remaining 0s are definite background
    
    return trimap


def refine_mask_with_vitmatte(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    trimap_width: int = 10,
    erode_radius: int = 2,
    dilate_radius: int = 2
) -> np.ndarray:
    """Refine a coarse mask using ViTMatte for high-quality alpha matte."""
    import torch
    
    vitmatte = load_vitmatte_model()
    model = vitmatte["model"]
    processor = vitmatte["processor"]
    device = vitmatte["device"]
    
    # Create trimap from coarse mask
    trimap = create_trimap(coarse_mask, trimap_width, erode_radius, dilate_radius)
    
    # Process inputs
    inputs = processor(images=image, trimaps=trimap, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        alpha = outputs.alphas[0, 0].cpu().numpy()
    
    return alpha


def run_point_bbox_inference(
    node,
    points: List[Dict[str, Any]],
    bbox: Optional[Tuple[float, float, float, float]]
) -> None:
    """Run SAM3 inference with points and/or bounding box prompts."""
    import torch
    import nuke
    
    try:
        # Load models
        sam_predictor = load_sam3_model()
        
        # Get image from input
        image = image_from_nuke_node(node)
        
        # Set image in predictor
        sam_predictor.set_image(image)
        
        # Prepare prompts
        point_coords = None
        point_labels = None
        box = None
        
        if points:
            point_coords = np.array([[p["x"], p["y"]] for p in points])
            point_labels = np.array([p["label"] for p in points])
        
        if bbox:
            box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
        
        # Run SAM3 prediction
        with torch.no_grad():
            masks, scores, logits = sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True
            )
        
        # Select best mask
        best_idx = np.argmax(scores)
        coarse_mask = masks[best_idx]
        
        # Get inference parameters
        from . import callbacks
        params = callbacks.get_inference_params(node)
        
        # Pre-processing
        coarse_mask = preprocess_mask(coarse_mask, params)
        
        # Refine with ViTMatte
        trimap_width = int(params["refiner_trimap_width"])
        erode_radius = int(params.get("trimap_erode_radius", 2))
        dilate_radius = int(params.get("trimap_dilate_radius", 2))
        if trimap_width > 0:
            alpha_matte = refine_mask_with_vitmatte(image, coarse_mask, trimap_width, erode_radius, dilate_radius)
        else:
            alpha_matte = coarse_mask.astype(np.float32)
        
        # Post-processing
        alpha_matte = postprocess_mask(alpha_matte, params)
        
        # Apply temporal consistency if enabled
        if params["enable_temporal_consistency"]:
            from . import temporal
            frame = nuke.frame()
            alpha_matte = temporal.apply_consistency(node, alpha_matte, frame, params)
        
        # Write result back to node
        write_mask_to_node(node, alpha_matte, params)
        
        print(f"[H2 SamViT] Inference complete. Score: {scores[best_idx]:.3f}")
        
    except Exception as e:
        import traceback
        nuke.message(f"Inference failed: {str(e)}\n\n{traceback.format_exc()}")


def run_text_prompt_inference(
    node,
    text_prompt: str,
    selection_points: List[Dict[str, Any]]
) -> None:
    """Run text-based object detection + SAM3 segmentation."""
    import torch
    import nuke
    
    try:
        # Load models
        text_detector = load_text_model()
        sam_predictor = load_sam3_model()
        
        # Get image
        image = image_from_nuke_node(node)
        
        # Run text-based detection
        model = text_detector["model"]
        processor = text_detector["processor"]
        device = text_detector["device"]
        
        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process detections
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=0.3,
            text_threshold=0.25,
            target_sizes=[(image.shape[0], image.shape[1])]
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        
        if len(boxes) == 0:
            nuke.message(f"No objects matching '{text_prompt}' found.")
            return
        
        # Select which detection to use
        if selection_points:
            # Use point to select the detection
            selected_idx = select_detection_by_point(boxes, selection_points[0])
        else:
            # Use highest scoring detection
            selected_idx = np.argmax(scores)
        
        selected_box = boxes[selected_idx]
        
        # Run SAM3 with detected box
        sam_predictor.set_image(image)
        
        with torch.no_grad():
            masks, mask_scores, logits = sam_predictor.predict(
                box=selected_box,
                multimask_output=True
            )
        
        best_idx = np.argmax(mask_scores)
        coarse_mask = masks[best_idx]
        
        # Continue with refinement (same as point/bbox)
        from . import callbacks
        params = callbacks.get_inference_params(node)
        
        coarse_mask = preprocess_mask(coarse_mask, params)
        
        trimap_width = int(params["refiner_trimap_width"])
        erode_radius = int(params.get("trimap_erode_radius", 2))
        dilate_radius = int(params.get("trimap_dilate_radius", 2))
        if trimap_width > 0:
            alpha_matte = refine_mask_with_vitmatte(image, coarse_mask, trimap_width, erode_radius, dilate_radius)
        else:
            alpha_matte = coarse_mask.astype(np.float32)
        
        alpha_matte = postprocess_mask(alpha_matte, params)
        
        if params["enable_temporal_consistency"]:
            from . import temporal
            frame = nuke.frame()
            alpha_matte = temporal.apply_consistency(node, alpha_matte, frame, params)
        
        write_mask_to_node(node, alpha_matte, params)
        
        print(f"[H2 SamViT] Text inference complete. Found {len(boxes)} objects, selected #{selected_idx + 1}")
        
    except Exception as e:
        import traceback
        nuke.message(f"Text inference failed: {str(e)}\n\n{traceback.format_exc()}")


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
    centers = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2 for box in boxes]
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
    
    # Final binary sharp
    if params["final_binary_sharp"]:
        mask = (mask > 0.5).astype(np.float32)
        # Apply slight blur for corner smoothing
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = (mask > 0.5).astype(np.float32)
    
    return np.clip(mask, 0, 1)


def write_mask_to_node(node, mask: np.ndarray, params: Dict[str, Any]) -> None:
    """Write the computed mask back to the Nuke node."""
    import nuke
    
    # Store mask in node's internal cache
    # In real implementation, this would write to the node's output channels
    
    # Cache the mask for later retrieval during render
    from . import cache
    frame = nuke.frame()
    node_name = node.name()
    
    cache.store_mask(node_name, frame, mask)
    
    # Force refresh the viewer
    nuke.updateUI()


def clear_models():
    """Clear loaded models to free GPU memory."""
    global _sam_model, _sam_predictor, _vitmatte_model, _text_model
    
    with _model_lock:
        _sam_model = None
        _sam_predictor = None
        _vitmatte_model = None
        _text_model = None
    
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("[H2 SamViT] Models cleared from memory.")
