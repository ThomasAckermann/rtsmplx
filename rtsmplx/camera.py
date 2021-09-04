import torch
import torch.nn as nn
import numpy as np
from rtsmplx.utils import transform_mat


class OrthographicCamera(nn.Module):
    """Orthographic camera model"""

    def __init__(self):
        super(OrthographicCamera, self).__init__()


        # register scale, rotation and translation parameters
        scale = torch.ones(1)
        rotation = torch.zeros(3)
        translation = torch.zeros((3,1))

        scale = nn.Parameter(scale, requires_grad=True)
        rotation = nn.Parameter(rotation, requires_grad=True)
        translation = nn.Parameter(translation, requires_grad=True)

        self.register_parameter("scale", scale)
        self.register_parameter("rotation", rotation)
        self.register_parameter("translation", translation)

    def forward(self, points):
        points_shape = points.shape
        points_reshape = torch.ones(points.shape[0], 4)
        points_reshape[:, :3] = points
        transform = transform_mat(self.rotation, self.translation, self.scale)
        projected_points = points_reshape @ transform.T
        projected_points = projected_points[:, :2]
        return projected_points

    def get_cam_transform(self):
        return transform_mat(self.rotation, self.translation, self.scale)


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

        rotation = torch.zeros(3)
        translation = torch.zeros(3)
        rotation = nn.Parameter(rotation, requires_grad=True)
        translation = nn.Parameter(translation, requires_grad=True)
        self.register_parameter("rotation", rotation)
        self.register_parameter("translation", translation)

    def forward(self):
        camera_mat = torch.zeros([self.batch_size, 2, 2])
        camera_mat[:, 0, 0] = self.focal_length_x
        camera_mat[:, 1, 1] = self.focal_length_y

        camera_transform = transform_mat(self.rotation, self.translation)

        projected_points = torch.einsum("ki,ji->jk", [camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2], projected_points[:, :, 2])
        img_points = torch.einsum("ki,ji->jk", [camera_mat, img_points])
        return img_points
    
