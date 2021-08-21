import torch
import numpy as np


class OrthographicCamera:
    """Orthographic camera model"""

    def __init__(self, bb_params=None, trans_mat=None):
        # bb_params: {0: right, 1: left, 2: top, 3: bottom, 4: far, 5: near}
        if bb_params == None:
            self.bb_params = torch.Tensor([1, 0, 1, 0, 1, 0])
        else:
            self.bb_params = bb_params

    def transformation_matrix(self):
        trans_mat = torch.Tensor(
            [
                [2 / (self.bb_params[0] - self.bb_params[1]), 0, 0, -1 * (self.bb_params [0] + self.bb_params[1]) / (self.bb_params[0] - self.bb_params[1])],
                [0, 2 / (self.bb_params[2] - self.bb_params[3]), 0, -1 * (self.bb_params[2] + self.bb_params[3]) / (self.bb_params[2] - self.bb_params[3])],
                [0, 0, -2 / (self.bb_params[4] - self.bb_params[5]), -1 * (self.bb_params[4] + self.bb_params[5]) / (self.bb_params[4] - self.bb_params[5])],
                [0, 0, 0, 1],
            ]
        )
        
        return trans_mat

    def orthographic_projection(self, points):
        points_shape = points.shape
        points_reshape = torch.ones(points.shape[0], 4)
        points_reshape[:, :3] = points
        
        trans_mat = self.transformation_matrix()
    
        projected_points = points_reshape @ trans_mat
        projected_points = projected_points[:, :2]
        return projected_points
