"""
Utility functions for mask processing and polygon conversion.

Hole Filling Strategy:
- SAM3 sometimes produces masks with multiple disconnected sections separated by tiny gaps
- This is common when objects have thin connecting parts or texture variations
- We use morphological closing (dilation followed by erosion) to bridge small gaps
- Then fill all interior holes using contour hierarchy analysis

Kernel sizes:
- kernel_size=5: Conservative, good for most cases
- kernel_size=7: Recommended default, bridges gaps up to ~7 pixels
- kernel_size=9+: More aggressive, may over-smooth fine details
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional

def simplify_mask_morphologically(mask: np.ndarray, simplification: float = 0.0) -> np.ndarray:
    """
    Simplify mask outline using morphological operations.
    
    This smooths the polygon by:
    - Removing small protrusions (like sticks pointing out)
    - Filling small gaps and indentations
    - Creating a cleaner, more generalized outline
    
    Args:
        mask: Binary mask to simplify
        simplification: 0.0 to 1.0 scale
                       0.0 = no simplification (original mask)
                       0.5 = moderate smoothing (removes small details)
                       1.0 = aggressive smoothing (removes protrusions, fills gaps)
    
    Returns:
        Simplified mask
    """
    if simplification <= 0.0:
        return mask
    
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    
    kernel_size = int(3 + simplification * 12)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if simplification < 0.5:
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    elif simplification < 0.8:
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        smoothed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        smoothed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return smoothed


def mask_to_polygon(mask: np.ndarray, simplification: float = 0.0) -> List[List[float]]:
    """
    Convert binary mask to polygon using contour detection.
    
    Args:
        mask: Binary mask to convert
        simplification: 0.0 to 1.0 scale for morphological simplification
                       0.0 = detailed outline (keeps all features)
                       0.5 = moderate smoothing (removes small protrusions)
                       1.0 = aggressive smoothing (removes details, fills gaps)
    
    Returns:
        Polygon as list of [x, y] coordinates
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255
    
    if simplification > 0:
        mask = simplify_mask_morphologically(mask, simplification)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    epsilon = 1.0
    largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(largest_contour) < 3:
        return []
    
    polygon = [[float(pt[0][0]), float(pt[0][1])] for pt in largest_contour]
    return polygon

def remove_small_regions(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """Remove small disconnected regions from mask."""
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    result = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 1
    
    return result

def fill_holes(mask: np.ndarray, kernel_size: int = 5, min_hole_area: int = 0) -> np.ndarray:
    """
    Fill holes in binary mask using morphological closing and hole filling.
    
    This handles multiple disconnected sections separated by small gaps by:
    1. First applying morphological closing to bridge tiny gaps
    2. Then filling all interior holes using contour hierarchy
    3. Optionally preserving large interior holes if min_hole_area > 0
    
    Args:
        mask: Binary mask with potential holes or small gaps
        kernel_size: Size of morphological kernel for closing small gaps (default: 5)
        min_hole_area: Minimum area of holes to preserve (default: 0, fills all holes)
    
    Returns:
        Mask with holes filled and small gaps bridged
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask
    
    filled = np.zeros_like(mask)
    
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(filled, [contour], -1, 1, -1)
        
        if min_hole_area > 0:
            for i, contour in enumerate(contours):
                if hierarchy[0][i][3] != -1:
                    area = cv2.contourArea(contour)
                    if area >= min_hole_area:
                        cv2.drawContours(filled, [contour], -1, 0, -1)
    else:
        cv2.drawContours(filled, contours, -1, 1, -1)
    
    return filled


def fill_holes_advanced(mask: np.ndarray, 
                       close_kernel_size: int = 7,
                       dilate_kernel_size: int = 3,
                       iterations: int = 1) -> np.ndarray:
    """
    Advanced hole filling with multiple morphological operations.
    
    This is more aggressive and useful for masks with many small disconnected regions.
    It performs:
    1. Morphological closing to bridge small gaps
    2. Optional dilation to further connect nearby regions
    3. Hole filling via contour hierarchy
    4. Erosion to restore approximate original size
    
    Args:
        mask: Binary mask with potential holes or small gaps
        close_kernel_size: Size of closing kernel (default: 7)
        dilate_kernel_size: Size of dilation/erosion kernel (default: 3)
        iterations: Number of dilation/erosion iterations (default: 1)
    
    Returns:
        Mask with holes filled and regions connected
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    
    if iterations > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
        dilated = cv2.dilate(closed, dilate_kernel, iterations=iterations)
    else:
        dilated = closed
    
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask
    
    filled = np.zeros_like(mask)
    
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(filled, [contour], -1, 1, -1)
    else:
        cv2.drawContours(filled, contours, -1, 1, -1)
    
    if iterations > 0:
        filled = cv2.erode(filled, dilate_kernel, iterations=iterations)
    
    return filled

def resolve_overlaps(masks: List[np.ndarray], scores: List[float]) -> List[np.ndarray]:
    """Resolve overlapping masks by keeping higher score for each pixel."""
    if not masks:
        return []
    
    h, w = masks[0].shape
    result = [np.zeros((h, w), dtype=np.uint8) for _ in masks]
    
    score_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    assigned = np.zeros((h, w), dtype=bool)
    
    for idx in score_indices:
        mask = masks[idx] > 0.5
        available = mask & ~assigned
        result[idx] = available.astype(np.uint8)
        assigned |= available
    
    return result

def nms_boxes(boxes: List[List[float]], scores: List[float], iou_threshold: float = 0.5) -> List[int]:
    """Apply Non-Maximum Suppression to boxes."""
    if not boxes:
        return []
    
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores_np.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def polygon_to_mask(polygon: List[List[float]], height: int, width: int) -> np.ndarray:
    """Convert polygon to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygon or len(polygon) < 3:
        return mask
    
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask


def polygon_iou(poly1: List[List[float]], poly2: List[List[float]], height: int, width: int) -> float:
    """Calculate IoU between two polygons using mask-based approach."""
    if not poly1 or not poly2 or len(poly1) < 3 or len(poly2) < 3:
        return 0.0
    
    mask1 = polygon_to_mask(poly1, height, width)
    mask2 = polygon_to_mask(poly2, height, width)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def filter_duplicate_polygons(
    new_polygons: List[List[List[float]]],
    existing_polygons: List[List[List[float]]],
    height: int,
    width: int,
    iou_threshold: float = 0.5
) -> List[int]:
    """
    Filter out new polygons that overlap significantly with existing ones.
    
    Returns indices of new_polygons that should be kept (don't overlap with existing).
    """
    if not new_polygons:
        return []
    
    if not existing_polygons:
        return list(range(len(new_polygons)))
    
    keep_indices = []
    
    for i, new_poly in enumerate(new_polygons):
        is_duplicate = False
        
        for existing_poly in existing_polygons:
            iou = polygon_iou(new_poly, existing_poly, height, width)
            if iou >= iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep_indices.append(i)
    
    return keep_indices
