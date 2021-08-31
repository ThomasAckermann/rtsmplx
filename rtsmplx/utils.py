import cv2
import math
import smplx
import torch
import numpy as np
import pyrender
import trimesh
import pytorch3d


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
    translation_mat = translation_matrix(translation)
    scale_mat_3 = scale * torch.eye(3)
    scale_mat = torch.eye(4)
    scale_mat[:3, :3] = scale_mat_2

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


def translation_matrix(transl):
    transl_matrix = torch.eye(4)
    transl_matrix[:, 3] = transl
    return transl_matrix
