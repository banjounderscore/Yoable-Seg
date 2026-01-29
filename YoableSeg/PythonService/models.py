from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class AutoLabelRequest(BaseModel):
    image_path: str
    prompts: List[str]
    text_threshold: float = 0.25
    box_threshold: float = 0.25
    max_instances: int = 50
    simplification: float = 0.0
    min_sam3_confidence: float = 0.6
    existing_polygons: Optional[List[List[List[float]]]] = None
    nms_iou_threshold: float = 0.5

class BoxToMaskRequest(BaseModel):
    image_path: str
    box: List[float]  # [x1, y1, x2, y2] in pixel coordinates
    simplification: float = 0.0
    min_sam3_confidence: float = 0.6
    existing_polygons: Optional[List[List[List[float]]]] = None
    nms_iou_threshold: float = 0.5

class RLEMask(BaseModel):
    counts: str
    size: List[int]  # [height, width]

class PolygonPoints(BaseModel):
    points: List[List[float]]  # [[x1, y1], [x2, y2], ...]

class Instance(BaseModel):
    box: List[float]  # [x1, y1, x2, y2]
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
