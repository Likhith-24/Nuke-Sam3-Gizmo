# filters.py - Image filters for mask post-processing
# Guided filter, bilateral filter, and edge enhancement

import numpy as np
import cv2
from typing import Optional


def box_filter(img: np.ndarray, radius: int) -> np.ndarray:
    """Apply box filter (mean filter)."""
    ksize = 2 * radius + 1
    return cv2.blur(img, (ksize, ksize))


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 8,
    eps: float = 0.01
) -> np.ndarray:
    """
    Apply guided filter for edge-preserving smoothing.
    
    The guided filter smooths the source image while preserving
    edges present in the guide image.
    
    Args:
        guide: Guide image (HxW, float32, 0-1)
        src: Source image to filter (HxW, float32, 0-1)
        radius: Filter radius
        eps: Regularization parameter (higher = more smoothing)
    
    Returns:
        Filtered image
    """
    # Ensure float32
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    
    # Mean of guide and source
    mean_I = box_filter(guide, radius)
    mean_p = box_filter(src, radius)
    
    # Correlation and variance
    mean_Ip = box_filter(guide * src, radius)
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = box_filter(guide * guide, radius)
    var_I = mean_II - mean_I * mean_I
    
    # Linear coefficients
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    # Mean of coefficients
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)
    
    # Output
    output = mean_a * guide + mean_b
    
    return np.clip(output, 0, 1)


def guided_filter_color(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 8,
    eps: float = 0.01
) -> np.ndarray:
    """
    Guided filter with color guide image.
    
    Args:
        guide: Color guide image (HxWx3, float32, 0-1)
        src: Source mask to filter (HxW, float32, 0-1)
        radius: Filter radius
        eps: Regularization parameter
    
    Returns:
        Filtered mask
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    
    # Split channels
    I_r = guide[:, :, 0]
    I_g = guide[:, :, 1]
    I_b = guide[:, :, 2]
    
    # Mean values
    mean_I_r = box_filter(I_r, radius)
    mean_I_g = box_filter(I_g, radius)
    mean_I_b = box_filter(I_b, radius)
    mean_p = box_filter(src, radius)
    
    # Covariance of guide and source
    mean_Ip_r = box_filter(I_r * src, radius)
    mean_Ip_g = box_filter(I_g * src, radius)
    mean_Ip_b = box_filter(I_b * src, radius)
    
    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p
    
    # Variance of guide (3x3 covariance matrix per pixel)
    var_I_rr = box_filter(I_r * I_r, radius) - mean_I_r * mean_I_r + eps
    var_I_rg = box_filter(I_r * I_g, radius) - mean_I_r * mean_I_g
    var_I_rb = box_filter(I_r * I_b, radius) - mean_I_r * mean_I_b
    var_I_gg = box_filter(I_g * I_g, radius) - mean_I_g * mean_I_g + eps
    var_I_gb = box_filter(I_g * I_b, radius) - mean_I_g * mean_I_b
    var_I_bb = box_filter(I_b * I_b, radius) - mean_I_b * mean_I_b + eps
    
    # Inverse of covariance matrix (simplified for efficiency)
    # Using element-wise operations rather than full matrix inverse
    
    # Compute a vector (simplified)
    inv_sum = 1.0 / (var_I_rr + var_I_gg + var_I_bb + eps)
    
    a_r = cov_Ip_r * inv_sum
    a_g = cov_Ip_g * inv_sum
    a_b = cov_Ip_b * inv_sum
    
    b = mean_p - a_r * mean_I_r - a_g * mean_I_g - a_b * mean_I_b
    
    # Mean of coefficients
    mean_a_r = box_filter(a_r, radius)
    mean_a_g = box_filter(a_g, radius)
    mean_a_b = box_filter(a_b, radius)
    mean_b = box_filter(b, radius)
    
    # Output
    output = mean_a_r * I_r + mean_a_g * I_g + mean_a_b * I_b + mean_b
    
    return np.clip(output, 0, 1)


def edge_aware_smooth(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    edge_strength: float = 1.0,
    smooth_strength: float = 0.5
) -> np.ndarray:
    """
    Edge-aware smoothing that preserves mask edges.
    
    Args:
        mask: Mask to smooth (HxW, 0-1)
        image: Optional reference image for edge detection
        edge_strength: How much to preserve edges (0-1)
        smooth_strength: Overall smoothing amount
    
    Returns:
        Smoothed mask
    """
    mask = mask.astype(np.float32)
    
    # Detect edges in mask
    mask_uint8 = (mask * 255).astype(np.uint8)
    edges = cv2.Canny(mask_uint8, 50, 150).astype(np.float32) / 255.0
    
    # Dilate edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # If image provided, also detect edges there
    if image is not None:
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        image_edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        image_edges = cv2.dilate(image_edges, kernel, iterations=2)
        
        # Combine edge maps
        edges = np.maximum(edges, image_edges * edge_strength)
    
    # Apply different smoothing based on edge presence
    smooth_radius = int(smooth_strength * 10) + 1
    smooth = cv2.GaussianBlur(mask, (smooth_radius * 2 + 1, smooth_radius * 2 + 1), 0)
    
    # Blend: edges keep original, non-edges get smoothed
    edge_weight = edges * edge_strength
    result = mask * edge_weight + smooth * (1 - edge_weight)
    
    return np.clip(result, 0, 1)


def morphological_smooth(
    mask: np.ndarray,
    iterations: int = 2
) -> np.ndarray:
    """
    Apply morphological operations for smoother mask edges.
    
    Args:
        mask: Input mask (0-1)
        iterations: Number of iterations
    
    Returns:
        Smoothed mask
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Opening (remove small noisy regions)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # Closing (fill small holes)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Gaussian blur for smooth edges
    smooth = cv2.GaussianBlur(closed, (5, 5), 0)
    
    return smooth.astype(np.float32) / 255.0


def refine_edges(
    mask: np.ndarray,
    image: np.ndarray,
    radius: int = 3
) -> np.ndarray:
    """
    Refine mask edges using image gradients.
    
    Args:
        mask: Input mask (0-1)
        image: Reference image (HxWx3 or HxW)
        radius: Edge refinement radius
    
    Returns:
        Edge-refined mask
    """
    mask = mask.astype(np.float32)
    
    # Convert image to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)
    
    gray = gray.astype(np.float32) / 255.0
    
    # Use guided filter for edge refinement
    refined = guided_filter(gray, mask, radius=radius, eps=0.01)
    
    return refined
