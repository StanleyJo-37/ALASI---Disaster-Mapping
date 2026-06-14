from pydantic import BaseModel, ConfigDict
import numpy as np

class OrthomosaicStitchingAlgorithm(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)
  
  images: np.ndarray = np.array([])
  
  def __init__(self, name, bases, dict, /, **kwds):
    super().__init__(name, bases, dict, **kwds)
  
  def __call__(self, *args, **kwds):
    
    return super().__call__(*args, **kwds)
