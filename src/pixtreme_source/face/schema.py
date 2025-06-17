import cupy as cp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Face(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: np.ndarray | cp.ndarray = Field(..., description="Bounding box in the format (x1, y1, x2, y2)")
    score: float = Field(..., description="Detection score")
    kps: np.ndarray | cp.ndarray | None = Field(default=None, description="Keypoints in the format (x, y)")
    matrix: np.ndarray | cp.ndarray | None = Field(default=None, description="Affine transformation matrix")
    embedding: np.ndarray | cp.ndarray | None = Field(default=None, description="Face embedding")
    normed_embedding: np.ndarray | cp.ndarray | None = Field(default=None, description="Normalized face embedding")
    image: np.ndarray | cp.ndarray | None = Field(default=None, description="Face image")
