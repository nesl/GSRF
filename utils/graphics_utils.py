
import torch
import math
import numpy as np
from typing import NamedTuple


# ---- point cloud data structure ----
class BasicPointCloud(NamedTuple):
    points : np.array
    attris : np.array
    normals : np.array
