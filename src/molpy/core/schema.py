import numpy as np
from pydantic import BaseModel, Field
import numpy.typing as npt
from numpydantic import NDArray, Shape

class BoxSchema(BaseModel):
    """
    Schema for a box in 3D space.
    """
    matrix: NDArray[Shape["3, 3"], float] = Field(..., description="3x3 matrix representing the box dimensions and angles")
    origin: NDArray[Shape["3"], float] = Field(..., description="Origin point of the box in 3D space")
    pbc: NDArray[Shape["3"], bool] = Field(..., description="Periodic boundary conditions in x, y, z directions")

