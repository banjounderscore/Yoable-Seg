"""
YoableSeg Python Inference Service
FastAPI server for SAM 3 segmentation with Grounding DINO

IMPLEMENTATION:

1. Grounding DINO: Text-prompted object detection
   - Input: Image + text prompts (e.g., "car. tire.")
   - Output: Bounding boxes + confidence scores + labels
   
2. Non-Maximum Suppression (NMS): Remove duplicate detections
   - Filters overlapping boxes based on IoU threshold
   
3. SAM3 Batch Inference: Generate precise segmentation masks
   - Input: Image + detected bounding boxes
   - Output: High-quality segmentation masks for each box
   
4. Post-processing: Refine masks
   - Remove small disconnected regions
   - Fill holes using morphological operations
   - Resolve overlapping masks (higher score wins)
   
5. Polygon Conversion: Convert masks to vector format
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
    BoxToMaskRequest, MaskResponse, Instance, HealthResponse
)
from utils import (
    mask_to_polygon, remove_small_regions,
    fill_holes, fill_holes_advanced, resolve_overlaps, nms_boxes,
    simplify_mask_morphologically, filter_duplicate_polygons, polygon_iou
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import Sam3TrackerProcessor, Sam3TrackerModel
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
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
        self.grounding_dino = None
        self.grounding_dino_processor = None
        self.sam3_model = None
        self.sam3_processor = None
        
    def load_grounding_dino(self):
        """Load GroundingDINO Base model."""
        if self.grounding_dino is not None:
            return
        
        try:
            model_id = "IDEA-Research/grounding-dino-base"
            logger.info(f"Loading GroundingDINO Base from cache: {model_id}")
            self.grounding_dino_processor = AutoProcessor.from_pretrained(
                model_id,
                local_files_only=True
            )
            self.grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id,
                local_files_only=True
            )
            self.grounding_dino.to(self.device)
            self.grounding_dino.eval()
            logger.info(f"GroundingDINO Base loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load GroundingDINO Base: {e}")
            logger.error("Make sure you ran the install script to download models.")
            raise

    def load_sam3(self):
        """Load SAM 3 Tracker model via Transformers for ultra-fast inference."""
        if self.sam3_model is not None:
            return
        
        if not SAM3_AVAILABLE:
            raise RuntimeError("Transformers SAM 3 not available. Install with: pip install transformers>=4.47.0")
        
        try:
            logger.info("Loading SAM 3 Tracker from cache: facebook/sam3")
            
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*sam3_video.*sam3_tracker.*")
                
                self.sam3_processor = Sam3TrackerProcessor.from_pretrained(
                    "facebook/sam3",
                    trust_remote_code=True,
                    local_files_only=True
                )
                self.sam3_model = Sam3TrackerModel.from_pretrained(
                    "facebook/sam3",
                    trust_remote_code=True,
                    ignore_mismatched_sizes=True,
                    local_files_only=True
                ).to(self.device)
            
            if self.device == "cuda":
                self.sam3_model = self.sam3_model.half()
                logger.info(f"SAM 3 Tracker loaded on {self.device} with FP16 - ultra-fast inference enabled")
            else:
                logger.info(f"SAM 3 Tracker loaded on {self.device}")
            
            self.sam3_model.eval()
            logger.info("SAM 3 Tracker ready for inference!")
            
        except Exception as e:
            logger.error(f"Failed to load SAM 3: {e}")
            logger.error("Make sure you ran the install script to download models.")
            raise

    def run_grounding_dino(
        self,
        image_path: str,
        prompts: List[str],
        text_threshold: float = 0.25,
        box_threshold: float = 0.25
    ) -> Tuple[np.ndarray, List[float], List[str]]:
        """
        Run GroundingDINO to get boxes from text prompts.
        
        Returns:
            boxes: np.ndarray of shape (N, 4) in xyxy format
            scores: List of confidence scores
            labels: List of text labels
        """
        self.load_grounding_dino()
        
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        
        prompts = [p if p.endswith(".") else p + "." for p in prompts]
        text_prompt = " ".join(prompts)
        
        inputs = self.grounding_dino_processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_dino(**inputs)
        
        results = self.grounding_dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)]
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy().tolist()
        labels = results["text_labels"]
        
        return boxes, scores, labels

    def run_sam3_with_box(
        self,
        image_path: str,
        box: List[float]
    ) -> Tuple[np.ndarray, float]:
        """
        Run SAM 3 Tracker with box prompt - finds the best mask that fits inside the box.
        
        Returns the mask that:
        1. Is at least 75% inside the provided box
        2. Is closest to the box size in terms of height/width
        """
        self.load_sam3()
        
        raw_image = Image.open(image_path).convert("RGB")
        img_width, img_height = raw_image.size
        
        input_boxes = [[[int(box[0]), int(box[1]), int(box[2]), int(box[3])]]]
        
        inputs = self.sam3_processor(
            images=raw_image,
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(self.device)
        
        # Get multiple mask candidates
        with torch.no_grad():
            outputs = self.sam3_model(**inputs, multimask_output=True)
        
        masks = self.sam3_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"]
        )[0]
        
        if masks.shape[0] == 0:
            return None, 0.0
        
        # Calculate box dimensions
        box_x1, box_y1, box_x2, box_y2 = box
        box_width = box_x2 - box_x1
        box_height = box_y2 - box_y1
        
        # Evaluate each mask candidate
        best_mask = None
        best_score = float('inf')  # Track minimum size difference
        best_iou_score = 0.0
        
        # Check the actual number of masks returned (dimension 1)
        num_masks = masks.shape[1] if len(masks.shape) > 1 else 1
        
        for i in range(num_masks):
            # Handle different mask shapes
            if len(masks.shape) == 4:  # [batch, num_masks, height, width]
                mask = masks[0, i].numpy().astype(np.uint8)
            elif len(masks.shape) == 3:  # [batch, height, width]
                mask = masks[0].numpy().astype(np.uint8) if i == 0 else None
                if mask is None:
                    break
            else:
                logger.error(f"Unexpected mask shape: {masks.shape}")
                return None, 0.0
            
            # Convert mask to polygon - this is what actually gets used
            polygon = mask_to_polygon(mask, simplification=0.0)
            
            if not polygon or len(polygon) < 3:
                continue
            
            # Calculate polygon bounding box
            polygon_array = np.array(polygon)
            poly_x_min = float(polygon_array[:, 0].min())
            poly_x_max = float(polygon_array[:, 0].max())
            poly_y_min = float(polygon_array[:, 1].min())
            poly_y_max = float(polygon_array[:, 1].max())
            
            mask_width = poly_x_max - poly_x_min
            mask_height = poly_y_max - poly_y_min
            
            # Calculate intersection between polygon bbox and input box
            intersect_left = max(poly_x_min, box_x1)
            intersect_right = min(poly_x_max, box_x2)
            intersect_top = max(poly_y_min, box_y1)
            intersect_bottom = min(poly_y_max, box_y2)
            
            intersect_width = max(0.0, intersect_right - intersect_left)
            intersect_height = max(0.0, intersect_bottom - intersect_top)
            intersect_area = intersect_width * intersect_height
            
            # Calculate polygon bbox area
            polygon_bbox_area = mask_width * mask_height
            
            # Calculate percentage of polygon bbox that's inside the input box
            percentage_inside = (intersect_area / polygon_bbox_area * 100.0) if polygon_bbox_area > 0 else 0.0
            
            # Get IoU score for this candidate
            candidate_iou_score = 0.0
            if outputs.iou_scores is not None and len(outputs.iou_scores.shape) == 3 and outputs.iou_scores.shape[2] > i:
                candidate_iou_score = float(outputs.iou_scores[0, 0, i])
            
            # Skip masks that aren't mostly inside the box (at least 75% contained)
            if percentage_inside < 75.0:
                continue
            
            # Calculate size difference: sum of width and height differences
            height_diff = abs(mask_height - box_height)
            width_diff = abs(mask_width - box_width)
            size_diff_sum = height_diff + width_diff
            
            # Select mask with smallest size difference
            if best_mask is None or size_diff_sum < best_score:
                best_score = size_diff_sum
                best_mask = mask
                best_iou_score = candidate_iou_score
        
        # If no mask met the criteria, return None
        if best_mask is None:
            return None, 0.0
        
        return best_mask, best_iou_score

    def run_sam3_with_boxes(
        self,
        image_path: str,
        boxes: List[List[float]]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Run SAM 3 Tracker with multiple box prompts - loops through run_sam3_with_box for each box."""
        result_masks = []
        result_scores = []
        
        for box in boxes:
            mask, score = self.run_sam3_with_box(image_path, box)
            if mask is not None:
                result_masks.append(mask)
                result_scores.append(score)
            else:
                result_masks.append(np.zeros((1, 1), dtype=np.uint8))
                result_scores.append(0.0)
        
        return result_masks, result_scores


model_manager = ModelManager()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "grounding_dino": model_manager.grounding_dino is not None,
            "sam3": model_manager.sam3_model is not None
        },
        gpu_available=torch.cuda.is_available()
    )


@app.post("/autolabel", response_model=AutoLabelResponse)
async def autolabel(request: AutoLabelRequest):
    """
    Run full auto-labeling pipeline:
    1. GroundingDINO for text-prompted object detection (get boxes, scores, labels)
    2. Apply NMS to remove overlapping detections
    3. Limit to max_instances
    4. SAM 3 batch inference for mask generation
    5. Post-process masks (remove small regions, fill holes)
    6. Resolve overlaps between masks
    7. Convert masks to polygons
    """
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
    
    try:
        image = Image.open(request.image_path)
        width, height = image.size
        
        boxes, scores, labels = model_manager.run_grounding_dino(
            request.image_path,
            request.prompts,
            request.text_threshold,
            request.box_threshold
        )
        
        if len(boxes) == 0:
            return AutoLabelResponse(width=width, height=height, instances=[])
        
        boxes_list = boxes.tolist()
        keep_indices = nms_boxes(boxes_list, scores, iou_threshold=0.5)
        
        boxes = boxes[keep_indices]
        scores = [scores[i] for i in keep_indices]
        labels = [labels[i] for i in keep_indices]
        
        if len(boxes) > request.max_instances:
            sorted_indices = np.argsort(scores)[::-1][:request.max_instances]
            boxes = boxes[sorted_indices]
            scores = [scores[i] for i in sorted_indices]
            labels = [labels[i] for i in sorted_indices]
        
        boxes_list = boxes.tolist()
        
        masks, mask_scores = model_manager.run_sam3_with_boxes(request.image_path, boxes_list)
        
        valid_masks = []
        valid_scores = []
        valid_boxes = []
        valid_labels = []
        
        for i, (mask, mask_score) in enumerate(zip(masks, mask_scores)):
            if mask_score < request.min_sam3_confidence:
                continue
                
            valid_masks.append(mask)
            valid_scores.append(scores[i])
            valid_boxes.append(boxes_list[i])
            valid_labels.append(labels[i])
        
        if len(valid_masks) == 0:
            return AutoLabelResponse(width=width, height=height, instances=[])
        
        processed_masks = []
        for mask in valid_masks:
            mask = remove_small_regions(mask, min_area=100)
            mask = fill_holes(mask, kernel_size=7)
            processed_masks.append(mask)
        
        final_masks = resolve_overlaps(processed_masks, valid_scores)
        
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
                'box': valid_boxes[i],
                'score': valid_scores[i],
                'phrase': valid_labels[i],
                'polygon': polygon
            })
        
        if request.existing_polygons and len(request.existing_polygons) > 0:
            keep_indices = filter_duplicate_polygons(
                new_polygons,
                request.existing_polygons,
                height,
                width,
                request.nms_iou_threshold
            )
            instance_data = [instance_data[i] for i in keep_indices]
        
        for data in instance_data:
            instances.append(Instance(
                box=data['box'],
                score=data['score'],
                phrase=data['phrase'],
                polygon=data['polygon'],
                rle=None
            ))
        
        return AutoLabelResponse(
            width=width,
            height=height,
            instances=instances
        )
        
    except Exception as e:
        logger.error(f"Auto-label error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/box_to_mask", response_model=MaskResponse)
async def box_to_mask(request: BoxToMaskRequest):
    """Generate mask from box prompt using SAM 3."""
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
    
    try:
        image = Image.open(request.image_path)
        width, height = image.size
        
        mask, score = model_manager.run_sam3_with_box(request.image_path, request.box)
        
        if mask is None:
            raise HTTPException(status_code=404, detail="No valid object found in box")
        
        if score < request.min_sam3_confidence:
            raise HTTPException(status_code=404, detail=f"Mask quality too low (score: {score:.2f}, required: {request.min_sam3_confidence:.2f})")
        
        mask = remove_small_regions(mask, min_area=50)
        mask = fill_holes(mask, kernel_size=7)
        
        polygon = mask_to_polygon(mask, simplification=request.simplification)
        
        if len(polygon) < 3:
            raise HTTPException(status_code=404, detail="No valid object found in box")
        
        if request.existing_polygons and len(request.existing_polygons) > 0:
            for existing_poly in request.existing_polygons:
                iou = polygon_iou(polygon, existing_poly, height, width)
                if iou >= request.nms_iou_threshold:
                    raise HTTPException(
                        status_code=409, 
                        detail=f"Label already exists (IoU: {iou:.2f} >= threshold: {request.nms_iou_threshold:.2f})"
                    )
        
        return MaskResponse(
            width=width,
            height=height,
            polygon=polygon,
            score=score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Box to mask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """
    Load SAM 3 model on startup for faster first inference.
    """
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    logger.info("=" * 60)
    logger.info("YoableSeg Inference Service")
    logger.info("Pipeline: Grounding DINO -> SAM3 -> Post-processing")
    logger.info("=" * 60)
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    logger.info(f"SAM 3 available: {SAM3_AVAILABLE}")
    logger.info("Loading SAM 3 Tracker model on startup...")
    try:
        model_manager.load_sam3()
        logger.info("SAM 3 Tracker loaded and ready - ultra-fast inference enabled")
        logger.info("Grounding DINO will be loaded on first use")
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
