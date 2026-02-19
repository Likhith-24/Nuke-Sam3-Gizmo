# temporal.py - Temporal consistency and smoothing for mask sequences
# Prevents mask flickering and jumping between frames

import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
import cv2


class TemporalBuffer:
    """Stores recent masks for temporal consistency calculations."""
    
    def __init__(self, max_frames: int = 10):
        self.max_frames = max_frames
        self._buffer: OrderedDict[int, np.ndarray] = OrderedDict()
    
    def add(self, frame: int, mask: np.ndarray) -> None:
        """Add a mask to the buffer."""
        self._buffer[frame] = mask.copy()
        
        # Remove oldest if over capacity
        while len(self._buffer) > self.max_frames:
            self._buffer.popitem(last=False)
    
    def get(self, frame: int) -> Optional[np.ndarray]:
        """Get mask for a specific frame."""
        return self._buffer.get(frame)
    
    def get_previous(self, frame: int) -> Optional[np.ndarray]:
        """Get the most recent mask before the given frame."""
        prev_frames = [f for f in self._buffer.keys() if f < frame]
        if not prev_frames:
            return None
        return self._buffer[max(prev_frames)]
    
    def get_range(self, start: int, end: int) -> Dict[int, np.ndarray]:
        """Get masks in a frame range."""
        return {f: m for f, m in self._buffer.items() if start <= f <= end}
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()


# Per-node temporal buffers
_temporal_buffers: Dict[str, TemporalBuffer] = {}


def get_temporal_buffer(node_name: str) -> TemporalBuffer:
    """Get or create temporal buffer for a node."""
    if node_name not in _temporal_buffers:
        _temporal_buffers[node_name] = TemporalBuffer()
    return _temporal_buffers[node_name]


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two masks."""
    # Binarize
    m1 = (mask1 > 0.5).astype(np.float32)
    m2 = (mask2 > 0.5).astype(np.float32)
    
    intersection = np.sum(m1 * m2)
    union = np.sum(m1) + np.sum(m2) - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def select_best_mask_temporal(
    candidate_masks: list,
    previous_mask: np.ndarray,
    temporal_weight: float = 0.5
) -> Tuple[np.ndarray, int]:
    """
    Select the best mask considering temporal consistency.
    
    Args:
        candidate_masks: List of (mask, score) tuples from SAM
        previous_mask: Mask from the previous frame
        temporal_weight: Weight for temporal IoU (0-1)
    
    Returns:
        Tuple of (best_mask, best_index)
    """
    if not candidate_masks:
        raise ValueError("No candidate masks provided")
    
    if previous_mask is None:
        # No previous frame - return highest scoring
        scores = [s for _, s in candidate_masks]
        best_idx = np.argmax(scores)
        return candidate_masks[best_idx][0], best_idx
    
    best_score = -1
    best_idx = 0
    
    for i, (mask, model_score) in enumerate(candidate_masks):
        # Compute temporal IoU
        iou = compute_iou(mask, previous_mask)
        
        # Combined score: (1 - weight) * model_score + weight * iou
        combined = (1 - temporal_weight) * model_score + temporal_weight * iou
        
        if combined > best_score:
            best_score = combined
            best_idx = i
    
    return candidate_masks[best_idx][0], best_idx


def suppress_jumping_regions(
    mask: np.ndarray,
    previous_mask: np.ndarray,
    threshold: float = 0.3
) -> np.ndarray:
    """
    Remove regions that suddenly appeared (likely false positives).
    
    Args:
        mask: Current frame mask
        previous_mask: Previous frame mask
        threshold: IoU threshold for suppression
    
    Returns:
        Filtered mask
    """
    if previous_mask is None:
        return mask
    
    # Binarize masks
    current_binary = (mask > 0.5).astype(np.uint8)
    prev_binary = (previous_mask > 0.5).astype(np.uint8)
    
    # Find connected components in current mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        current_binary, connectivity=8
    )
    
    # Check each component
    filtered_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels):  # Skip background (0)
        component_mask = (labels == i).astype(np.float32)
        
        # Check overlap with previous mask
        overlap = np.sum(component_mask * prev_binary)
        component_area = np.sum(component_mask)
        
        if component_area == 0:
            continue
        
        overlap_ratio = overlap / component_area
        
        # Keep component if it has sufficient overlap with previous frame
        # or if it's a significant portion of the total mask
        total_area = np.sum(current_binary)
        is_significant = (component_area / total_area) > 0.1 if total_area > 0 else False
        
        if overlap_ratio >= threshold or is_significant:
            filtered_mask += mask * component_mask
    
    return np.clip(filtered_mask, 0, 1)


def temporal_smooth(
    mask: np.ndarray,
    previous_masks: Dict[int, np.ndarray],
    smoothing_strength: float = 0.5,
    edges_only: bool = False,
    edge_width: int = 8
) -> np.ndarray:
    """
    Apply temporal smoothing across frames.
    
    Args:
        mask: Current frame mask
        previous_masks: Dict of frame -> mask for recent frames
        smoothing_strength: Blending strength (0-1)
        edges_only: Only smooth edge regions (motion-aware)
        edge_width: Width of edge region when edges_only=True
    
    Returns:
        Smoothed mask
    """
    if not previous_masks or smoothing_strength == 0:
        return mask
    
    # Sort frames
    sorted_frames = sorted(previous_masks.keys())
    
    # Weighted average with exponential decay
    weights = []
    masks_to_blend = [mask]
    weight_sum = 1.0
    
    for i, frame in enumerate(reversed(sorted_frames)):
        # Exponential decay
        age = i + 1
        weight = smoothing_strength ** age
        weights.append(weight)
        masks_to_blend.append(previous_masks[frame])
        weight_sum += weight
    
    # Normalize weights
    weights = [1.0 / weight_sum] + [w / weight_sum for w in weights]
    
    if edges_only:
        # Create edge mask
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        
        # Detect edges
        kernel = np.ones((edge_width, edge_width), np.uint8)
        dilated = cv2.dilate(mask_binary, kernel, iterations=1)
        eroded = cv2.erode(mask_binary, kernel, iterations=1)
        edge_mask = ((dilated - eroded) > 0).astype(np.float32)
        
        # Smooth edge region only
        blended = np.zeros_like(mask)
        for m, w in zip(masks_to_blend, weights):
            # Resize if needed
            if m.shape != mask.shape:
                m = cv2.resize(m, (mask.shape[1], mask.shape[0]))
            blended += m * w
        
        # Blend: use smoothed in edge region, original elsewhere
        result = mask * (1 - edge_mask) + blended * edge_mask
    else:
        # Full frame smoothing
        result = np.zeros_like(mask)
        for m, w in zip(masks_to_blend, weights):
            if m.shape != mask.shape:
                m = cv2.resize(m, (mask.shape[1], mask.shape[0]))
            result += m * w
    
    return np.clip(result, 0, 1)


def apply_consistency(
    node,
    mask: np.ndarray,
    frame: int,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Apply full temporal consistency pipeline.
    
    Args:
        node: Nuke node
        mask: Current frame mask
        frame: Current frame number
        params: Inference parameters
    
    Returns:
        Temporally consistent mask
    """
    node_name = node.name()
    buffer = get_temporal_buffer(node_name)
    
    result = mask.copy()
    
    # Layer 1: Temporal IoU-based mask selection (if multiple candidates)
    # This is handled in inference.py when selecting from SAM outputs
    
    # Layer 2: Suppress jumping regions
    if params.get("enable_temporal_consistency", True):
        previous = buffer.get_previous(frame)
        if previous is not None:
            threshold = params.get("suppression_threshold", 30) / 100.0
            result = suppress_jumping_regions(result, previous, threshold)
    
    # Layer 3: Temporal smoothing
    if params.get("enable_temporal", False):
        smoothing = params.get("temporal_smoothing", 50) / 100.0
        edges_only = params.get("smooth_edges_only", False)
        edge_width = int(params.get("edge_width", 8))
        
        # Get recent masks
        recent = buffer.get_range(frame - 5, frame - 1)
        
        if recent:
            result = temporal_smooth(
                result,
                recent,
                smoothing_strength=smoothing,
                edges_only=edges_only,
                edge_width=edge_width
            )
    
    # Store in buffer for future frames
    buffer.add(frame, result)
    
    return result


def clear_temporal_buffer(node_name: str) -> None:
    """Clear temporal buffer for a node."""
    if node_name in _temporal_buffers:
        _temporal_buffers[node_name].clear()
        print(f"[Temporal] Buffer cleared for {node_name}")


def clear_all_temporal_buffers() -> None:
    """Clear all temporal buffers."""
    for buffer in _temporal_buffers.values():
        buffer.clear()
    print("[Temporal] All buffers cleared")
