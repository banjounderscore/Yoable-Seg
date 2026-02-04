"""
YoableSeg Python Inference Service
FastAPI server for SAM 3 segmentation with text prompts

IMPLEMENTATION:

1. SAM3 Text-Prompted Segmentation: Detect and segment using text prompts
   - Input: Image + text prompts (e.g., "person")
   - Output: Masks, bounding boxes, confidence scores
   - Supports negative prompts to exclude certain objects
   
2. Non-Maximum Suppression (NMS): Remove duplicate detections
   - Filters overlapping masks based on IoU threshold
   
3. Post-processing: Refine masks
   - Remove small disconnected regions
   - Fill holes using morphological operations
   - Resolve overlapping masks (higher score wins)
   
4. Polygon Conversion: Convert masks to vector format
   - Extract contours from masks
   - Simplify polygons for efficient storage


Run with: uvicorn inference_server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, List, Tuple
import logging
import cv2

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    AutoLabelRequest, AutoLabelResponse,
    MaskResponse, Instance, HealthResponse,
    LoopToMaskRequest
)
from utils import (
    mask_to_polygon, remove_small_regions,
    fill_holes, fill_holes_advanced, resolve_overlaps,
    simplify_mask_morphologically, filter_duplicate_polygons, polygon_iou,
    polygon_to_mask, compute_polygon_match_score, point_in_polygon,
    clip_polygon_to_region, detect_and_fill_boundary_gaps,
    get_loop_bounding_box, expand_box, combine_bounding_boxes,
    find_best_matching_segment, create_special_case_polygon
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import Sam3Processor, Sam3Model
    from transformers import Sam3TrackerProcessor, Sam3TrackerModel
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    logger.error("Transformers SAM 3 not available - install with: pip install transformers>=4.47.0")

app = FastAPI(title="YoableSeg Inference Service - SAM 3", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam3_model = None
        self.sam3_processor = None
        self.sam3_tracker_model = None
        self.sam3_tracker_processor = None
        
    def load_sam3(self):
        """Load SAM 3 model for text-prompted segmentation."""
        if self.sam3_model is not None:
            return
        
        if not SAM3_AVAILABLE:
            raise RuntimeError("Transformers SAM 3 not available. Install with: pip install transformers>=4.47.0")
        
        try:
            logger.info("Loading SAM 3 from cache: facebook/sam3")
            
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*sam3.*")
                
                self.sam3_processor = Sam3Processor.from_pretrained(
                    "facebook/sam3",
                    trust_remote_code=True
                )
                self.sam3_model = Sam3Model.from_pretrained(
                    "facebook/sam3",
                    trust_remote_code=True,
                    ignore_mismatched_sizes=True
                ).to(self.device)
            
            logger.info(f"SAM 3 loaded on {self.device} - text-prompt segmentation enabled")
            
            self.sam3_model.eval()
            logger.info("SAM 3 ready for text-prompted inference!")
            
        except Exception as e:
            logger.error(f"Failed to load SAM 3: {e}")
            logger.error("Make sure you ran the install script to download models.")
            raise

    def load_sam3_tracker(self):
        """Load SAM 3 Tracker model for box-prompted segmentation."""
        if self.sam3_tracker_model is not None:
            return
        
        if not SAM3_AVAILABLE:
            raise RuntimeError("Transformers SAM 3 not available. Install with: pip install transformers>=4.47.0")
        
        try:
            logger.info("Loading SAM 3 Tracker from cache: facebook/sam3")
            
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*sam3_video.*sam3_tracker.*")
                
                self.sam3_tracker_processor = Sam3TrackerProcessor.from_pretrained(
                    "facebook/sam3",
                    trust_remote_code=True
                )
                self.sam3_tracker_model = Sam3TrackerModel.from_pretrained(
                    "facebook/sam3",
                    trust_remote_code=True,
                    ignore_mismatched_sizes=True
                ).to(self.device)
            
            logger.info(f"SAM 3 Tracker loaded on {self.device}")
            
            self.sam3_tracker_model.eval()
            logger.info("SAM 3 Tracker ready for box-prompted inference!")
            
        except Exception as e:
            logger.error(f"Failed to load SAM 3 Tracker: {e}")
            logger.error("Make sure you ran the install script to download models.")
            raise

    def run_sam3_text_prompt(
        self,
        image: Image.Image,
        prompt: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Run SAM 3 with text prompt for concept segmentation.
        
        Args:
            image: PIL Image
            prompt: Text prompt (e.g., "person")
            threshold: Score threshold for detections
            mask_threshold: Threshold for binarizing masks
        
        Returns:
            masks: List of numpy arrays, each [H, W] binary mask
            scores: List of confidence scores
        """
        self.load_sam3()
        
        inputs = self.sam3_processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sam3_model(**inputs)
        
        original_sizes = inputs.get("original_sizes")
        if original_sizes is not None:
            target_sizes = original_sizes.tolist()
        else:
            width, height = image.size
            target_sizes = [(height, width)]
        
        results = self.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=target_sizes
        )[0]
        
        masks_tensor = results.get("masks", None)
        scores_tensor = results.get("scores", None)
        
        if masks_tensor is None or len(masks_tensor) == 0:
            logger.info(f"No detections for prompt '{prompt}'")
            return [], []
        
        masks = []
        for m in masks_tensor:
            if isinstance(m, torch.Tensor):
                mask_np = m.cpu().numpy()
            else:
                mask_np = np.array(m)
            masks.append(mask_np.astype(np.uint8))
        
        scores = []
        if scores_tensor is not None:
            if isinstance(scores_tensor, torch.Tensor):
                scores = scores_tensor.cpu().numpy().tolist()
            else:
                scores = list(scores_tensor)
        
        logger.info(f"Found {len(masks)} objects for prompt '{prompt}'")
        return masks, scores

    def detect_negative_masks(
        self,
        image: Image.Image,
        negative_prompts: List[str],
        threshold: float = 0.3
    ) -> List[np.ndarray]:
        """
        Detect objects matching negative prompts and return their masks.
        These masks will be subtracted from positive detections.
        """
        all_masks = []
        
        for prompt in negative_prompts:
            logger.info(f"Detecting negative concept: '{prompt}'")
            masks, _ = self.run_sam3_text_prompt(
                image,
                prompt,
                threshold=threshold,
                mask_threshold=0.5
            )
            all_masks.extend(masks)
            logger.info(f"Found {len(masks)} masks for negative prompt '{prompt}'")
        
        return all_masks
    
    def subtract_negative_from_mask(
        self,
        positive_mask: np.ndarray,
        negative_masks: List[np.ndarray]
    ) -> Tuple[np.ndarray, bool]:
        """
        Subtract negative masks from a positive mask.
        
        Returns:
            modified_mask: The mask with negative regions removed
            should_keep: False if the mask is mostly negative content
        """
        if not negative_masks:
            return positive_mask, True
        
        positive_area = positive_mask.sum()
        if positive_area == 0:
            return positive_mask, False
        
        combined_negative = np.zeros_like(positive_mask)
        for neg_mask in negative_masks:
            if neg_mask.shape == positive_mask.shape:
                combined_negative = np.logical_or(combined_negative, neg_mask > 0)
        
        overlap = np.logical_and(positive_mask > 0, combined_negative)
        overlap_area = overlap.sum()
        overlap_ratio = overlap_area / positive_area
        
        if overlap_ratio > 0.8:
            return positive_mask, False
        
        modified_mask = positive_mask.copy()
        modified_mask[combined_negative > 0] = 0
        
        if modified_mask.sum() < 100:
            return positive_mask, False
        
        return modified_mask, True


model_manager = ModelManager()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "sam3": model_manager.sam3_model is not None,
            "sam3_tracker": model_manager.sam3_tracker_model is not None
        },
        gpu_available=torch.cuda.is_available()
    )


@app.post("/autolabel", response_model=AutoLabelResponse)
async def autolabel(request: AutoLabelRequest):
    """
    Run full auto-labeling pipeline using SAM3 text prompts:
    1. Run each positive prompt independently to get masks
    2. Optionally detect negative prompts and subtract from positive masks
    3. Post-process masks (remove small regions, fill holes)
    4. Resolve mask overlaps
    5. Convert masks to polygons
    6. Filter against existing polygons using polygon IoU
    """
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
    
    try:
        image = Image.open(request.image_path).convert("RGB")
        width, height = image.size
        
        logger.info(f"Processing image: {request.image_path} ({width}x{height})")
        logger.info(f"Positive prompts: {request.prompts}")
        logger.info(f"Negative prompts: {request.negative_prompts}")
        logger.info(f"Score threshold: {request.score_threshold}, Mask threshold: {request.mask_threshold}")
        
        negative_masks = []
        if request.negative_prompts and len(request.negative_prompts) > 0:
            logger.info(f"Detecting negative concepts: {request.negative_prompts}")
            negative_masks = model_manager.detect_negative_masks(
                image,
                request.negative_prompts,
                threshold=request.score_threshold
            )
            logger.info(f"Found {len(negative_masks)} negative masks to exclude")
        
        all_masks = []
        all_scores = []
        all_labels = []
        
        for prompt in request.prompts:
            logger.info(f"Running detection for prompt: '{prompt}'")
            masks, scores = model_manager.run_sam3_text_prompt(
                image,
                prompt,
                threshold=request.score_threshold,
                mask_threshold=request.mask_threshold
            )
            
            logger.info(f"Prompt '{prompt}': found {len(masks)} raw instances")
            
            kept_count = 0
            for mask, score in zip(masks, scores):
                if negative_masks:
                    modified_mask, should_keep = model_manager.subtract_negative_from_mask(
                        mask, negative_masks
                    )
                    if not should_keep:
                        logger.info(f"  Filtered out detection (too much overlap with negative)")
                        continue
                    mask = modified_mask
                
                all_masks.append(mask)
                all_scores.append(score)
                all_labels.append(prompt.lower())
                kept_count += 1
            
            logger.info(f"Prompt '{prompt}': kept {kept_count} instances after negative filtering")
        
        if len(all_masks) == 0:
            logger.info("No detections found for any prompt")
            return AutoLabelResponse(width=width, height=height, instances=[])
        
        logger.info(f"Total detections: {len(all_masks)}")
        
        if len(all_masks) > request.max_instances:
            sorted_indices = np.argsort(all_scores)[::-1][:request.max_instances]
            all_masks = [all_masks[i] for i in sorted_indices]
            all_scores = [all_scores[i] for i in sorted_indices]
            all_labels = [all_labels[i] for i in sorted_indices]
            logger.info(f"Limited to top {request.max_instances} instances")
        
        processed_masks = []
        for mask in all_masks:
            mask = remove_small_regions(mask, min_area=100)
            mask = fill_holes(mask, kernel_size=7)
            processed_masks.append(mask)
        
        final_masks = resolve_overlaps(processed_masks, all_scores)
        
        instances = []
        new_polygons = []
        instance_data = []
        
        for i in range(len(final_masks)):
            mask = final_masks[i]
            if mask.sum() < 100:
                continue
            
            polygon = mask_to_polygon(mask, simplification=request.simplification)
            if len(polygon) < 3:
                continue
            
            new_polygons.append(polygon)
            instance_data.append({
                'score': all_scores[i],
                'phrase': all_labels[i],
                'polygon': polygon
            })
        
        logger.info(f"Generated {len(new_polygons)} valid polygons")
        
        if request.existing_polygons and len(request.existing_polygons) > 0:
            logger.info(f"Checking against {len(request.existing_polygons)} existing polygons (IoU threshold: {request.nms_iou_threshold})")
            keep_indices = filter_duplicate_polygons(
                new_polygons,
                request.existing_polygons,
                height,
                width,
                request.nms_iou_threshold
            )
            filtered_count = len(instance_data) - len(keep_indices)
            instance_data = [instance_data[i] for i in keep_indices]
            new_polygons = [new_polygons[i] for i in keep_indices]
            logger.info(f"Filtered {filtered_count} duplicates, {len(instance_data)} remain")
        
        for data in instance_data:
            instances.append(Instance(
                score=data['score'],
                phrase=data['phrase'],
                polygon=data['polygon'],
                rle=None
            ))
        
        logger.info(f"Returning {len(instances)} instances")
        
        return AutoLabelResponse(
            width=width,
            height=height,
            instances=instances
        )
        
    except Exception as e:
        logger.error(f"Auto-label error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/loop_to_mask", response_model=MaskResponse)
async def loop_to_mask(request: LoopToMaskRequest):
    """
    Generate mask from freehand loop selection using SAM3 polygon matching.
    
    Algorithm:
    1. Get bounding box of user loop and expand it
    2. Use SAM3 Tracker with multimask_output=True to get multiple candidate masks
    3. For each candidate, compute area-based score: |polygon outside loop| + |loop outside polygon|
    4. Select polygon with minimum score
    5. Special case: If loop starts/ends inside same existing label:
       - Restrict search to polygons outside that label
       - Apply only the portion inside loop and outside existing label
       - Fill gaps along boundary to ensure continuity
    """
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
    
    try:
        from PIL import Image
        
        image = Image.open(request.image_path).convert("RGB")
        width, height = image.size
        
        logger.info(f"Processing loop selection for: {request.image_path} ({width}x{height})")
        logger.info(f"Loop has {len(request.loop_points)} points")
        
        # Check special case: loop starts and ends inside same existing label
        is_special_case = False
        starting_label_polygon = None
        starting_label_mask = None
        
        if request.starting_label_id is not None and request.existing_labels:
            if 0 <= request.starting_label_id < len(request.existing_labels):
                starting_label_polygon = request.existing_labels[request.starting_label_id]
                starting_label_mask = polygon_to_mask(starting_label_polygon, height, width)
                
                # Check if loop starts AND ends inside this label
                loop_start = tuple(request.loop_points[0])
                loop_end = tuple(request.loop_points[-1])
                
                start_inside = point_in_polygon(loop_start, starting_label_polygon)
                end_inside = point_in_polygon(loop_end, starting_label_polygon)
                
                if start_inside and end_inside:
                    is_special_case = True
                    logger.info(f"Special case detected: loop starts/ends inside label {request.starting_label_id}")
        
        # Get bounding box for SAM3 query
        # For both normal and special case: just use the loop
        search_bbox = get_loop_bounding_box(request.loop_points)
        if is_special_case:
            logger.info(f"Special case: Loop bbox: {search_bbox}")
        else:
            logger.info(f"Normal case: Loop bbox: {search_bbox}")
        
        # Use moderate expansion (20%) to focus on the object while including some context
        # Too much expansion (e.g., 50%) includes excessive background and reduces SAM3 quality
        expanded_bbox = expand_box(
            search_bbox, 
            expansion_factor=0.2,
            image_width=width,
            image_height=height
        )
        
        logger.info(f"Search bbox expanded: {expanded_bbox} (image: {width}x{height})")
        
        # Use SAM3 Tracker to get multiple candidate masks
        logger.info("Querying SAM3 Tracker for candidate masks...")
        model_manager.load_sam3_tracker()
        
        # Prepare input - ensure coordinates are within bounds
        input_boxes = [[[
            max(0, int(expanded_bbox[0])), 
            max(0, int(expanded_bbox[1])), 
            min(width, int(expanded_bbox[2])), 
            min(height, int(expanded_bbox[3]))
        ]]]
        
        # Use ONLY the box prompt (no points)
        # Providing both box and points simultaneously confuses SAM3 and produces low-quality masks
        # The loop points are used for:
        # 1. Defining the bounding box (already done above)
        # 2. Matching against candidate masks AFTER generation (done later)
        inputs = model_manager.sam3_tracker_processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(model_manager.device)
        
        # Use multimask_output=True to get multiple candidate masks
        # SAM3 Tracker returns exactly 3 masks with multimask_output=True (fixed behavior)
        # representing different segmentation hypotheses at varying confidence levels
        with torch.no_grad():
            outputs = model_manager.sam3_tracker_model(**inputs, multimask_output=True)
        
        masks = model_manager.sam3_tracker_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"]
        )[0]
        
        if masks.shape[0] == 0:
            raise HTTPException(status_code=404, detail="No candidate masks found")
        
        logger.info(f"SAM3 Tracker returned masks with shape: {masks.shape}")
        
        # Extract candidate polygons from masks
        # Filter by SAM score BEFORE computing expensive match scores
        candidates = []
        num_masks = masks.shape[1] if len(masks.shape) > 1 else 1
        
        for i in range(num_masks):
            if len(masks.shape) == 4:
                mask = masks[0, i].numpy().astype(np.uint8)
            elif len(masks.shape) == 3:
                mask = masks[0].numpy().astype(np.uint8) if i == 0 else None
                if mask is None:
                    break
            else:
                logger.error(f"Unexpected mask shape: {masks.shape}")
                break
            
            # Get IoU score from SAM3
            sam_score = 1.0
            if outputs.iou_scores is not None and len(outputs.iou_scores.shape) == 3:
                if outputs.iou_scores.shape[2] > i:
                    sam_score = float(outputs.iou_scores[0, 0, i])
            
            logger.info(f"  Mask {i}: SAM score {sam_score:.3f}")
            
            # Filter by quality threshold BEFORE processing
            if sam_score < request.min_sam3_confidence:
                logger.info(f"  Mask {i} rejected: SAM score {sam_score:.3f} < threshold {request.min_sam3_confidence:.3f}")
                continue
            
            # Clean up mask
            mask = remove_small_regions(mask, min_area=100)
            mask = fill_holes(mask, kernel_size=5)
            
            # Convert to polygon
            polygon = mask_to_polygon(mask, simplification=0.0)
            
            if len(polygon) >= 3:
                candidates.append((polygon, sam_score, mask))
        
        logger.info(f"Found {len(candidates)} candidate polygons from masks (after quality filtering)")
        
        if len(candidates) == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"No masks met quality threshold (SAM score >= {request.min_sam3_confidence:.3f}). Try drawing a more precise loop or lowering the quality threshold in settings."
            )
        
        # Handle special case with segment-based matching
        if is_special_case:
            logger.info("Applying special case segment-based matching...")
            
            # Convert candidates to list of polygons for vicinity search
            candidate_polygons = [poly for poly, _, _ in candidates]
            
            # Find the best matching segment
            best_match = find_best_matching_segment(
                candidate_polygons,
                request.loop_points,
                starting_label_mask,
                height,
                width
            )
            
            if best_match is None:
                logger.warning("No matching segment found, falling back to original logic")
                # Fallback to original filtering
                filtered_candidates = []
                
                for polygon, sam_score, mask in candidates:
                    # Check if polygon is fully contained inside existing label
                    overlap = np.logical_and(mask, starting_label_mask)
                    poly_area = np.sum(mask)
                    overlap_area = np.sum(overlap)
                    
                    # Skip polygons fully inside the existing label
                    if poly_area > 0 and overlap_area / poly_area > 0.95:
                        continue
                    
                    # Check if polygon intersects the user loop
                    loop_mask = polygon_to_mask(request.loop_points, height, width)
                    loop_poly_overlap = np.logical_and(mask, loop_mask)
                    
                    if np.sum(loop_poly_overlap) > 0:
                        filtered_candidates.append((polygon, sam_score, mask))
                
                logger.info(f"After fallback filtering: {len(filtered_candidates)} candidates remain")
                candidates = filtered_candidates
                
                if len(candidates) == 0:
                    raise HTTPException(status_code=404, detail="No valid polygons found outside existing label")
            else:
                poly_idx, segment_indices, match_score = best_match
                logger.info(f"Found best matching segment: polygon {poly_idx}, score {match_score:.2f}")
                
                # Use the best matching polygon for final processing
                # Filter candidates to only include the best match
                best_polygon = candidate_polygons[poly_idx]
                best_mask = candidates[poly_idx][2]
                best_sam_score = candidates[poly_idx][1]
                
                candidates = [(best_polygon, best_sam_score, best_mask)]
                logger.info(f"Selected best matching segment from polygon {poly_idx}")
        
        # Compute match scores for all candidates
        logger.info("Computing match scores...")
        scored_candidates = []
        
        for polygon, sam_score, mask in candidates:
            match_score = compute_polygon_match_score(polygon, request.loop_points, height, width)
            scored_candidates.append((polygon, match_score, sam_score, mask))
        
        # Select best match (minimum match score)
        # Note: All candidates already passed quality threshold
        scored_candidates.sort(key=lambda x: x[1])
        best_polygon, best_match_score, best_sam_score, best_mask = scored_candidates[0]
        
        logger.info(f"Selected best polygon with match score: {best_match_score:.2f}, SAM score: {best_sam_score:.3f}")
        
        # Apply polygon based on case
        if is_special_case:
            logger.info("Applying special case label application...")
            
            # Create masks
            loop_mask = polygon_to_mask(request.loop_points, height, width)
            
            # Clip polygon to: inside loop AND outside existing label
            inside_loop = np.logical_and(best_mask, loop_mask)
            outside_existing = np.logical_and(inside_loop, ~starting_label_mask)
            
            # Detect and fill gaps along boundary
            final_mask = detect_and_fill_boundary_gaps(outside_existing.astype(np.uint8), starting_label_mask)
            
            # Alternative: Create special case polygon that follows the label border
            # This connects exit and entry points by following the label boundary
            # special_polygon = create_special_case_polygon(
            #     request.loop_points,
            #     starting_label_polygon,
            #     starting_label_mask,
            #     height,
            #     width
            # )
            # final_mask = polygon_to_mask(special_polygon, height, width)
            
            logger.info(f"Applied clipped polygon with gap filling")
        else:
            # Normal case: use full selected polygon
            final_mask = best_mask.astype(np.uint8)
        
        # Post-process the final mask
        final_mask = remove_small_regions(final_mask, min_area=50)
        final_mask = fill_holes(final_mask, kernel_size=5)
        
        # Convert to polygon
        final_polygon = mask_to_polygon(final_mask, simplification=request.simplification)
        
        if len(final_polygon) < 3:
            raise HTTPException(status_code=404, detail="Final polygon has too few points")
        
        # NMS: Check against existing labels to avoid duplicates
        if request.existing_labels and len(request.existing_labels) > 0:
            nms_threshold = 0.5  # IoU threshold for duplicate detection
            
            for idx, existing_polygon in enumerate(request.existing_labels):
                # Skip the starting label in special case (it will be removed by C# code)
                if is_special_case and request.starting_label_id is not None and idx == request.starting_label_id:
                    continue
                
                iou = polygon_iou(final_polygon, existing_polygon, height, width)
                if iou > nms_threshold:
                    logger.info(f"Loop polygon rejected: IoU {iou:.3f} with existing label {idx} (threshold: {nms_threshold})")
                    raise HTTPException(
                        status_code=409, 
                        detail=f"Loop polygon too similar to existing label (IoU: {iou:.3f})"
                    )
            
            logger.info(f"Loop polygon passed NMS check against {len(request.existing_labels)} existing labels")
        
        logger.info(f"Returning polygon with {len(final_polygon)} points")
        
        return MaskResponse(
            width=width,
            height=height,
            polygon=final_polygon,
            score=best_sam_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Loop to mask error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """
    Load SAM 3 models on startup for faster first inference.
    """
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    logger.info("=" * 60)
    logger.info("YoableSeg Inference Service")
    logger.info("Pipeline: SAM3 Text Prompts -> Post-processing")
    logger.info("=" * 60)
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    logger.info(f"SAM 3 available: {SAM3_AVAILABLE}")
    logger.info("Loading SAM 3 model on startup...")
    try:
        model_manager.load_sam3()
        logger.info("SAM 3 loaded - text-prompt segmentation enabled")
        logger.info("SAM 3 Tracker will be loaded on first box-prompt use")
        logger.info("Ready to accept requests!")
    except Exception as e:
        logger.error(f"Failed to preload SAM 3 model: {e}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info("=" * 60)
    logger.info("Starting YoableSeg Inference Server")
    logger.info(f"Server: {host}:{port}")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    logger.info("=" * 60)
    
    uvicorn.run(app, host=host, port=port)
