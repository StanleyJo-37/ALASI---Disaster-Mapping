from typing import TypedDict, List

import numpy as np

class YOLOSegmentationTypedDict(TypedDict):
  """
    Attributes:
      class_ids: 1D-array of class ids of each instance.
      bboxes: 2D-array containing the bbox of each instance.
              The format is [[x, y, w, h]]
      masks: 2D-array containing the jagged coordinates of each instance.
             The format is [[x1, y1, x2, y2]]
  """
  
  class_ids: List[int]
  bboxes: List[np.array]
  masks: List[np.array]