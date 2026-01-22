from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class VectorAnnotation:
    slice_index: int
    contours_px: List[np.ndarray]  # each (N, 2)
    label: str                     # e.g. "calcium"
    artery_name: Optional[str] = None


@dataclass
class AnnotationBundle:
    vectors: Optional[Dict[int, List[VectorAnnotation]]] = None
    masks: Optional[np.ndarray] = None
    mask_labels: Optional[Dict[int, str]] = None
