import cv2
import math
import smplx
import torch
import numpy as np
import pyrender
import trimesh
import pytorch3d
import io

import whatimage
import pyheif
from PIL import Image


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def video_capture(path, framerate=5):
    video_capture = cv2.VideoCapture(path)
    framerate = video_capture.get(framerate)
    count = 1
    while video_capture.isOpened():
        frame_id = video_capture.get(1)
        ret, frame = video_capture.read()
        if ret != True:
            break
        if frame_id % math.floor(framerate) == 0:
            filename = "frame%d.jpg" % count
            count += 1
            cv2.imwrite(filename, frame)
    video_capture.release()
    return "Done"


def rot_mat_2d(angle):
    s = torch.sin(angle)
    c = torch.cos(angle)
    return torch.Tensor([[c, -s], [s, c]])


def angle_between(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    rotation_angle = torch.Tensor([np.arccos(np.dot(vec1, vec2))])
    return rotation_angle


def transform_mat(rot, transl, scale=1):
    # translation matrix
    eye_3 = torch.eye(3)
    bottom = torch.Tensor([[0.0, 0.0, 0.0, 1.0]])
    transl_mat_3 = torch.cat((eye_3, transl), dim=1)
    translation_mat = torch.cat((transl_mat_3, bottom), dim=0)

    # scale matrix
    scale_mat_3 = scale * torch.eye(3)
    scale_mat = torch.eye(4)
    scale_mat[:3, :3] = scale_mat_3

    # rotation matrix
    rotation_mat_3 = pytorch3d.transforms.axis_angle_to_matrix(rot)
    rotation_mat = torch.eye(4)
    rotation_mat[:3, :3] = rotation_mat_3

    transform = translation_mat @ scale_mat @ rotation_mat

    return transform


def transform_mat_persp(rot, transl):
    # translation matrix
    eye_3 = torch.eye(3)
    bottom = torch.Tensor([[0.0, 0.0, 0.0, 1.0]])
    transl_mat_3 = torch.cat((eye_3, transl), dim=1)
    translation_mat = torch.cat((transl_mat_3, bottom), dim=0)

    # rotation matrix
    rotation_mat_3 = pytorch3d.transforms.axis_angle_to_matrix(rot)
    rotation_mat = torch.eye(4)
    rotation_mat[:3, :3] = rotation_mat_3

    transform = translation_mat @ rotation_mat

    return transfor



def get_torch_trans_format(translation, rot_angles):
    rotation_mat_3 = pytorch3d.transforms.axis_angle_to_matrix(rot)
    return (translation, rotation_mat)


def get_fov():
    
