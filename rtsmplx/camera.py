import torch
import numpy as np


class Camera:
    def __init__(self):
        pass

    def orthographic_projection(self, points):
        projected_points = points[:,:2]
        return projected_points


        
