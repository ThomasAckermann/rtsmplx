import torch
import torch.nn as nn
import numpy as np
from rtsmplx.utils import transform_mat
from rtsmplx.utils import transform_mat_persp
import rtsmplx.utils as utils
import pytorch3d


class OrthographicCamera(nn.Module):
    """Orthographic camera model"""

    def __init__(self):
        super(OrthographicCamera, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # register scale, rotation and translation parameters
        scale = torch.ones(3, device=self.device)
        rotation = torch.zeros(3, device=self.device)
        translation = torch.zeros((3,1), device=self.device)

        scale = nn.Parameter(scale, requires_grad=True)
        rotation = nn.Parameter(rotation, requires_grad=True)
        translation = nn.Parameter(translation, requires_grad=True)

        self.register_parameter("scale", scale)
        self.register_parameter("rotation", rotation)
        self.register_parameter("translation", translation)

    def forward(self, points):
        points_shape = points.shape
        points_reshape = torch.ones([points.shape[0], 4], device=self.device)
        points_reshape[:, :3] = points
        transform = transform_mat(self.rotation, self.translation, scale=self.scale, device=self.device)
        projected_points = points_reshape @ transform.T
        projected_points = projected_points[:, :2]
        return projected_points

    def get_cam_transform(self, cpu=False):
        if cpu:
            return transform_mat(self.rotation, self.translation, device="cpu")
        else:
            return transform_mat(self.rotation, self.translation, device=self.device)


class PerspectiveCamera(nn.Module):
    """Perspective camera model"""

    FOCAL_LENGTH = 1
    def __init__(self):
        super(PerspectiveCamera, self).__init__()

    def __init__(self, focal_length_x=None, focal_length_y=None):
        super(PerspectiveCamera, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # register rotation and translation parameters
        rotation = torch.zeros(3)
        translation = torch.zeros((3,1))

        rotation = nn.Parameter(rotation, requires_grad=True)
        translation = nn.Parameter(translation, requires_grad=True)

        self.register_parameter("rotation", rotation)
        self.register_parameter("translation", translation)

        # register focal length

        focal_length_x = None
        focal_length_y = None
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

        # register center
        center = torch.zeros([2])
        center = nn.Parameter(center, requires_grad=True)
        self.register_parameter('center', center)


    def forward(self, points):
        points_shape = points.shape
        points_reshape = torch.ones([points.shape[0], 4], device=self.device)
        points_reshape[:, :3] = points
        camera_mat = torch.zeros([2, 2], device=self.device)
        camera_mat[0, 0] = self.focal_length_x
        camera_mat[1, 1] = self.focal_length_y

        camera_transform = transform_mat(self.rotation, self.translation, device=self.device)

        projected_points = torch.einsum("ki,ji->jk", [camera_transform, points_reshape]).to(device=self.device)

        img_points = torch.div(projected_points[:, :2], projected_points[:, 2][:,None])
        img_points = torch.einsum("ki,ji->jk", [camera_mat, img_points])
        return img_points

    def get_cam_transform(self, cpu=False):
        if cpu:
            return transform_mat(self.rotation, self.translation, device="cpu")
        else:
            return transform_mat(self.rotation, self.translation, device=self.device)



class OrthographicCameraTorch(nn.Module):
    """Orthographic camera model"""

    def __init__(self):
        super(OrthographicCameraTorch, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # register scale, rotation and translation parameters
        scale = torch.ones((1, 3), device=self.device)
        rotation = torch.randn((1, 3), device=self.device)
        translation = torch.randn((1, 3), device=self.device)
        frustum_max = torch.ones(3, device=self.device)
        frustum_min = torch.ones(3, device=self.device) * -1

        frustum_max[0] = 100.
        frustum_max[0] = 1.

        scale = nn.Parameter(scale, requires_grad=True)
        rotation = nn.Parameter(rotation, requires_grad=True)
        translation = nn.Parameter(translation, requires_grad=True)
        frustum_max = nn.Parameter(frustum_max, requires_grad=True)
        frustum_min = nn.Parameter(frustum_min, requires_grad=True)

        self.register_parameter("scale", scale)
        self.register_parameter("rotation", rotation)
        self.register_parameter("translation", translation)
        self.register_parameter("frustum_max", frustum_max)
        self.register_parameter("frustum_min", frustum_min)


    def get_transform(self):
        rotation_mat = pytorch3d.transforms.axis_angle_to_matrix(self.rotation)
        return pytorch3d.transforms.Transform3d().rotate(rotation_mat).scale(self.scale).translate(self.translation)


    def forward(self, points, image_size=[512, 512]):
        points_shape = points.shape
        points_reshape = points.reshape((1,points.shape[0], 3)).to(device=self.device)
        rotation_mat = pytorch3d.transforms.axis_angle_to_matrix(self.rotation).to(device=self.device)
        render_camera = pytorch3d.renderer.cameras.FoVOrthographicCameras(
                R=rotation_mat,
                T=self.translation,
                device=self.device,
                scale_xyz=self.scale,
                zfar=self.frustum_max[0],
                znear=self.frustum_min[0],
                max_y=self.frustum_max[1],
                min_y=self.frustum_min[1],
                max_x=self.frustum_max[2],
                min_x=self.frustum_min[2],
                )
        # transform = pytorch3d.transforms.Transform3d().rotate(rotation_mat).scale(self.scale).translate(self.translation).to(device=self.device)
        projected_points = render_camera.transform_points(points_reshape, image_size=image_size)
        # projected_points = transform.transform_points(points_reshape).reshape(points_shape[0], 3)[:,:2]
        projected_points = projected_points.reshape(points_shape[0], 3)[:,:2]
        return projected_points

