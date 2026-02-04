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


def polygon_area(polygon: List[List[float]]) -> float:
    """Calculate the area of a polygon using the shoelace formula."""
    if not polygon or len(polygon) < 3:
        return 0.0
    
    area = 0.0
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def polygon_area_outside_loop(polygon: List[List[float]], loop: List[List[float]], 
                               height: int, width: int) -> float:
    """Calculate the area of polygon that lies outside the user loop using Shapely."""
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    
    try:
        poly_shape = Polygon([(pt[0], pt[1]) for pt in polygon])
        loop_shape = Polygon([(pt[0], pt[1]) for pt in loop])
        
        # Ensure polygons are valid
        if not poly_shape.is_valid:
            poly_shape = make_valid(poly_shape)
        if not loop_shape.is_valid:
            loop_shape = make_valid(loop_shape)
        
        # Area of polygon NOT in loop = polygon - (polygon ∩ loop)
        outside = poly_shape.difference(loop_shape)
        return float(outside.area)
    except Exception as e:
        logger.warning(f"Shapely polygon operation failed: {e}, falling back to mask method")
        # Fallback to mask method if Shapely fails
        poly_mask = polygon_to_mask(polygon, height, width)
        loop_mask = polygon_to_mask(loop, height, width)
        outside = np.logical_and(poly_mask, ~loop_mask)
        return float(np.sum(outside))


def loop_area_outside_polygon(loop: List[List[float]], polygon: List[List[float]], 
                               height: int, width: int) -> float:
    """Calculate the area of user loop that lies outside the polygon using Shapely."""
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    
    try:
        poly_shape = Polygon([(pt[0], pt[1]) for pt in polygon])
        loop_shape = Polygon([(pt[0], pt[1]) for pt in loop])
        
        # Ensure polygons are valid
        if not poly_shape.is_valid:
            poly_shape = make_valid(poly_shape)
        if not loop_shape.is_valid:
            loop_shape = make_valid(loop_shape)
        
        # Area of loop NOT in polygon = loop - (loop ∩ polygon)
        outside = loop_shape.difference(poly_shape)
        return float(outside.area)
    except Exception as e:
        logger.warning(f"Shapely polygon operation failed: {e}, falling back to mask method")
        # Fallback to mask method if Shapely fails
        poly_mask = polygon_to_mask(polygon, height, width)
        loop_mask = polygon_to_mask(loop, height, width)
        outside = np.logical_and(loop_mask, ~poly_mask)
        return float(np.sum(outside))


def compute_polygon_match_score(polygon: List[List[float]], loop: List[List[float]], 
                                 height: int, width: int) -> float:
    """
    Compute area-based matching score for polygon selection using pure polygon operations.
    score = |polygon outside loop| + |loop outside polygon|
    Lower score = better match.
    
    Uses Shapely for precise polygon operations without mask conversion.
    """
    a_out = polygon_area_outside_loop(polygon, loop, height, width)
    a_in = loop_area_outside_polygon(loop, polygon, height, width)
    return a_out + a_in


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        
        j = i
    
    return inside


def clip_polygon_to_region(polygon: List[List[float]], region_mask: np.ndarray) -> np.ndarray:
    """
    Clip a polygon to only include the portion inside a region mask.
    Returns a binary mask of the clipped polygon.
    """
    poly_mask = polygon_to_mask(polygon, region_mask.shape[0], region_mask.shape[1])
    clipped = np.logical_and(poly_mask, region_mask)
    return clipped.astype(np.uint8)


def detect_and_fill_boundary_gaps(polygon_mask: np.ndarray, existing_label_mask: np.ndarray) -> np.ndarray:
    """
    Detect and fill gaps along the boundary where polygon meets existing label.
    
    This ensures continuity by filling minimal regions required to connect
    the polygon segment to the external edge of the existing label.
    
    Args:
        polygon_mask: Binary mask of the selected polygon segment (inside loop, outside existing label)
        existing_label_mask: Binary mask of the existing label
    
    Returns:
        Binary mask with gaps filled
    """
    # Find the boundary of the existing label (external edge)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    existing_dilated = cv2.dilate(existing_label_mask, kernel, iterations=1)
    existing_boundary = existing_dilated - existing_label_mask
    
    # Find the boundary of the polygon segment
    polygon_dilated = cv2.dilate(polygon_mask, kernel, iterations=1)
    polygon_boundary = polygon_dilated - polygon_mask
    
    # Find regions where boundaries are close but not touching
    # These are potential gaps that need filling
    close_proximity = cv2.dilate(polygon_boundary, kernel, iterations=2)
    potential_gaps = np.logical_and(close_proximity, existing_boundary)
    
    if not np.any(potential_gaps):
        return polygon_mask  # No gaps to fill
    
    # Fill the gaps by connecting nearby regions
    # Use morphological closing to bridge small gaps
    combined = np.logical_or(polygon_mask, existing_label_mask)
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filled_combined = cv2.morphologyEx(combined.astype(np.uint8), cv2.MORPH_CLOSE, closing_kernel)
    
    # Extract only the new filled regions (not part of original existing label)
    new_filled = np.logical_and(filled_combined, ~existing_label_mask)
    
    return new_filled.astype(np.uint8)


def get_loop_bounding_box(loop: List[List[float]]) -> List[float]:
    """Get bounding box of a loop/polygon in [x1, y1, x2, y2] format."""
    xs = [p[0] for p in loop]
    ys = [p[1] for p in loop]
    return [min(xs), min(ys), max(xs), max(ys)]


def combine_bounding_boxes(bbox1: List[float], bbox2: List[float]) -> List[float]:
    """
    Combine two bounding boxes into a single bounding box that encompasses both.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
    
    Returns:
        Combined bounding box [x1, y1, x2, y2]
    """
    return [
        min(bbox1[0], bbox2[0]),  # x1
        min(bbox1[1], bbox2[1]),  # y1
        max(bbox1[2], bbox2[2]),  # x2
        max(bbox1[3], bbox2[3])   # y2
    ]


def expand_box(box: List[float], expansion_factor: float = 0.25, 
               image_width: Optional[int] = None, 
               image_height: Optional[int] = None) -> List[float]:
    """
    Expand a bounding box by a factor.
    
    Args:
        box: [x1, y1, x2, y2] bounding box
        expansion_factor: Factor to expand the box by
        image_width: Optional image width to clamp x coordinates
        image_height: Optional image height to clamp y coordinates
    
    Returns:
        Expanded bounding box [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    expand_w = width * expansion_factor
    expand_h = height * expansion_factor
    
    expanded = [
        max(0, x1 - expand_w),
        max(0, y1 - expand_h),
        x2 + expand_w,
        y2 + expand_h
    ]
    
    # Clamp to image bounds if provided
    if image_width is not None:
        expanded[0] = max(0, expanded[0])
        expanded[2] = min(image_width, expanded[2])
    
    if image_height is not None:
        expanded[1] = max(0, expanded[1])
        expanded[3] = min(image_height, expanded[3])
    
    return expanded


def get_polygons_in_vicinity(all_polygons: List[List[List[float]]], 
                              loop_bbox: List[float], 
                              expansion: float = 50.0) -> List[int]:
    """
    Get indices of polygons that are in the vicinity of the loop bounding box.
    
    Args:
        all_polygons: List of all available polygons
        loop_bbox: Bounding box of the loop [x1, y1, x2, y2]
        expansion: How much to expand the search area (in pixels)
    
    Returns:
        List of indices of polygons within the vicinity
    """
    x1, y1, x2, y2 = loop_bbox
    x1 -= expansion
    y1 -= expansion
    x2 += expansion
    y2 += expansion
    
    vicinity_indices = []
    for idx, polygon in enumerate(all_polygons):
        # Check if any vertex of the polygon is within the expanded bbox
        for point in polygon:
            if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                vicinity_indices.append(idx)
                break
    
    return vicinity_indices


def segment_polygon_by_label(polygon: List[List[float]], 
                              label_mask: np.ndarray) -> List[List[int]]:
    """
    Divide polygon vertices into segments that are outside the label.
    
    Each segment is a consecutive sequence of vertices that are all outside the label.
    Vertices inside the label act as segment boundaries.
    
    Args:
        polygon: List of [x, y] coordinates
        label_mask: Binary mask of the label
    
    Returns:
        List of segments, where each segment is a list of vertex indices
    """
    if not polygon or len(polygon) < 2:
        return []
    
    height, width = label_mask.shape
    
    # Check which vertices are outside the label
    outside_status = []
    for point in polygon:
        x, y = int(round(point[0])), int(round(point[1]))
        # Clamp to image bounds
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        
        is_outside = (label_mask[y, x] == 0)
        outside_status.append(is_outside)
    
    # Find consecutive segments of vertices that are outside
    segments = []
    current_segment = []
    
    for i, is_outside in enumerate(outside_status):
        if is_outside:
            current_segment.append(i)
        else:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
    
    # Don't forget the last segment
    if current_segment:
        # Check if this segment wraps around to the beginning
        if segments and outside_status[0]:
            # Merge with the first segment (wraps around)
            segments[0] = current_segment + segments[0]
        else:
            segments.append(current_segment)
    
    return segments


def compute_segment_match_score(polygon: List[List[float]], 
                                segment_indices: List[int],
                                loop_points: List[List[float]],
                                loop_mask: np.ndarray,
                                height: int, 
                                width: int) -> float:
    """
    Compute match score for a polygon segment against the loop.
    
    The score is: |segment area outside loop| + |loop area outside segment|
    Lower score = better match.
    
    Args:
        polygon: Full polygon coordinates
        segment_indices: Indices of vertices in this segment
        loop_points: User-drawn loop coordinates
        loop_mask: Binary mask of the loop
        height: Image height
        width: Image width
    
    Returns:
        Match score (lower is better)
    """
    if not segment_indices or len(segment_indices) < 2:
        return float('inf')
    
    # Create a mask for just this segment by drawing edges between consecutive vertices
    segment_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw the segment edges
    for i in range(len(segment_indices)):
        idx1 = segment_indices[i]
        idx2 = segment_indices[(i + 1) % len(segment_indices)] if i < len(segment_indices) - 1 else segment_indices[0]
        
        # Only draw edge if consecutive in the original polygon
        if idx2 == (idx1 + 1) % len(polygon):
            pt1 = (int(round(polygon[idx1][0])), int(round(polygon[idx1][1])))
            pt2 = (int(round(polygon[idx2][0])), int(round(polygon[idx2][1])))
            cv2.line(segment_mask, pt1, pt2, 1, thickness=2)
    
    # Also create a small region around the segment for area calculations
    # Dilate the segment edges to create a thin mask region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    segment_region = cv2.dilate(segment_mask, kernel, iterations=2)
    
    # Compute areas:
    # 1. Segment region outside loop
    outside_loop = np.logical_and(segment_region, ~loop_mask)
    area_outside = float(np.sum(outside_loop))
    
    # 2. Loop outside segment region
    outside_segment = np.logical_and(loop_mask, ~segment_region)
    area_inside = float(np.sum(outside_segment))
    
    return area_outside + area_inside


def find_best_matching_segment(all_polygons: List[List[List[float]]],
                                loop_points: List[List[float]],
                                label_mask: np.ndarray,
                                height: int,
                                width: int) -> Optional[Tuple[int, List[int], float]]:
    """
    Find the polygon segment that best matches the loop while outside the label.
    
    This implements the special case logic:
    1. Get polygons in vicinity of the loop
    2. For each polygon, divide vertices into segments that are outside the label
    3. For each segment, compute match score
    4. Return the best matching segment
    
    Args:
        all_polygons: List of all available polygons
        loop_points: User-drawn loop coordinates
        label_mask: Binary mask of the starting label
        height: Image height
        width: Image width
    
    Returns:
        Tuple of (polygon_index, segment_vertex_indices, score) or None if no match found
    """
    if not all_polygons or not loop_points:
        return None
    
    # Create loop mask
    loop_mask = polygon_to_mask(loop_points, height, width)
    
    # Get loop bounding box to find nearby polygons
    loop_bbox = get_loop_bounding_box(loop_points)
    vicinity_indices = get_polygons_in_vicinity(all_polygons, loop_bbox)
    
    if not vicinity_indices:
        return None
    
    best_match = None
    best_score = float('inf')
    
    # For each polygon in vicinity
    for poly_idx in vicinity_indices:
        polygon = all_polygons[poly_idx]
        
        # Divide polygon into segments outside the label
        segments = segment_polygon_by_label(polygon, label_mask)
        
        # For each segment, compute match score
        for segment_indices in segments:
            if len(segment_indices) < 2:
                continue
            
            score = compute_segment_match_score(
                polygon, 
                segment_indices, 
                loop_points, 
                loop_mask,
                height, 
                width
            )
            
            if score < best_score:
                best_score = score
                best_match = (poly_idx, segment_indices, score)
    
    return best_match


def create_special_case_polygon(loop_points: List[List[float]],
                                label_polygon: List[List[float]],
                                label_mask: np.ndarray,
                                height: int,
                                width: int) -> List[List[float]]:
    """
    Create polygon for special case: loop that starts inside label, goes outside, returns inside.
    
    The final polygon consists of:
    1. The portion of the loop that's outside the label
    2. The label border segments that connect the exit and entry points
    
    Args:
        loop_points: User-drawn loop coordinates
        label_polygon: The polygon of the starting/ending label
        label_mask: Binary mask of the starting label
        height: Image height
        width: Image width
    
    Returns:
        Final polygon coordinates
    """
    if not loop_points or not label_polygon:
        return loop_points
    
    # Find where loop exits and enters the label
    exit_points = []
    enter_points = []
    
    for i in range(len(loop_points) - 1):
        pt1 = loop_points[i]
        pt2 = loop_points[i + 1]
        
        x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
        x2, y2 = int(round(pt2[0])), int(round(pt2[1]))
        
        # Clamp to bounds
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        
        inside1 = label_mask[y1, x1] > 0
        inside2 = label_mask[y2, x2] > 0
        
        # Transition from inside to outside = exit point
        if inside1 and not inside2:
            exit_points.append(i + 1)  # The point that's outside
        
        # Transition from outside to inside = enter point
        if not inside1 and inside2:
            enter_points.append(i + 1)  # The point that's inside
    
    # If we found exit and entry points, construct the final polygon
    if exit_points and enter_points:
        # Take the first exit and last enter
        exit_idx = exit_points[0]
        enter_idx = enter_points[-1]
        
        # The loop portion outside the label
        outside_portion = loop_points[exit_idx:enter_idx]
        
        # Find closest points on label border to connect
        # For simplicity, we'll use the loop points at exit/enter as connection points
        # In a more sophisticated implementation, you'd find the actual intersection points
        
        # Create the final polygon: outside portion + border connection
        final_polygon = outside_portion.copy()
        
        # Add border connection (simplified - just close the polygon)
        # In reality, you'd trace along the label border
        if final_polygon and len(final_polygon) >= 3:
            return final_polygon
    
    # Fallback: just return the loop
    return loop_points
