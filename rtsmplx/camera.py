import torch
import torch.nn as nn
import numpy as np


class OrthographicCamera(nn.Module):
    """Orthographic camera model"""

    def __init__(self, bb_params=None, trans_mat=None):
        super(OrthographicCamera, self).__init__()
        # cam_param = torch.Tensor([1, 0, 1, 0, 1, 0])
        cam_param = torch.eye(4)
        cam_param = nn.Parameter(cam_param, requires_grad=True)
        self.register_parameter("cam_param", cam_param)
        # bb_params: {0: right, 1: left, 2: top, 3: bottom, 4: far, 5: near}
        if bb_params == None:
            self.bb_params = torch.Tensor([1, 0, 1, 0, 1, 0])
        else:
            self.bb_params = bb_params

    def transformation_matrix(self):
        trans_mat = torch.Tensor(
            [
                [
                    2 / (self.cam_param[0] - self.cam_param[1]),
                    0,
                    0,
                    -1
                    * (self.cam_param[0] + self.cam_param[1])
                    / (self.cam_param[0] - self.cam_param[1]),
                ],
                [
                    0,
                    2 / (self.cam_param[2] - self.cam_param[3]),
                    0,
                    -1
                    * (self.cam_param[2] + self.cam_param[3])
                    / (self.cam_param[2] - self.cam_param[3]),
                ],
                [
                    0,
                    0,
                    -2 / (self.cam_param[4] - self.cam_param[5]),
                    -1
                    * (self.cam_param[4] + self.cam_param[5])
                    / (self.cam_param[4] - self.cam_param[5]),
                ],
                [0, 0, 0, 1],
            ]
        )

        return trans_mat

    def forward(self, points):
        """
        np_points = points.detach().numpy()
        self.bb_params = [
            np.max(np_points[:, 0]),
            np.min(np_points[:, 0]),
            np.max(np_points[:, 1]),
            np.min(np_points[:, 1]),
            np.max(np_points[:, 2]),
            np.min(np_points[:, 2]),
        ]
        """
        points_shape = points.shape
        points_reshape = torch.ones(points.shape[0], 4)
        points_reshape[:, :3] = points

        # trans_mat = self.transformation_matrix()
        trans_mat = self.cam_param

        projected_points = points_reshape @ trans_mat
        projected_points = projected_points[:, :2]
        return projected_points


class PerspectiveCamera(nn.Module):
    FOCAL_LENGTH = 5000

    def __init__(self, focal_length_x=None, focal_length_y=None):
        super(OrthographicCamera, self).__init__()

        focal_length_x = torch.Tensor(
            [self.FOCAL_LENGTH if focal_length_x is None else focal_length_x]
        )
        focal_length_y = torch.Tensor(
            [self.FOCAL_LENGTH if focal_length_y is None else focal_length_y]
        )

        focal_length_x = nn.Parameter(focal_length_x, requires_grad=True)
        focal_length_y = nn.Parameter(focal_length_y, requires_grad=True)
        self.register_parameter("focal_length_x", focal_length_x)
        self.register_parameter("focal_length_y", focal_length_y)


    def forward(self):

