from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class AutoLabelRequest(BaseModel):
    image_path: str
    prompts: List[str]
    negative_prompts: Optional[List[str]] = None
    score_threshold: float = 0.5
    mask_threshold: float = 0.5
    max_instances: int = 50
    simplification: float = 0.0
    existing_polygons: Optional[List[List[List[float]]]] = None
    nms_iou_threshold: float = 0.5

class RLEMask(BaseModel):
    counts: str
    size: List[int]  # [height, width]

class PolygonPoints(BaseModel):
    points: List[List[float]]  # [[x1, y1], [x2, y2], ...]

class Instance(BaseModel):
    score: float
    phrase: Optional[str] = None
    polygon: List[List[float]]  # [[x1, y1], [x2, y2], ...]
    rle: Optional[RLEMask] = None

class AutoLabelResponse(BaseModel):
    width: int
    height: int
    instances: List[Instance]

class MaskResponse(BaseModel):
    width: int
    height: int
    polygon: List[List[float]]
    score: float = 1.0

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    gpu_available: bool

class LoopToMaskRequest(BaseModel):
    image_path: str
    loop_points: List[List[float]]  # [[x1, y1], [x2, y2], ...] user-drawn loop
    existing_labels: Optional[List[List[List[float]]]] = None  # existing label polygons
    starting_label_id: Optional[int] = None  # ID of label where loop started (if inside one)
    simplification: float = 0.0
    min_sam3_confidence: float = 0.6
