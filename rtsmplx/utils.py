import cv2
import math
import smplx
import torch
import numpy as np
import pyrender
import trimesh


def video_capture(path, framerate=5):
    video_capture = cv2.VideoCapture(path)
    framerate = video_capture.get(framerate)
    count = 1
    while (video_capture.isOpened()):
        frame_id = video_capture.get(1)
        ret, frame = video_capture.read()
        if (ret != True):
            break
        if (frame_id % math.floor(framerate) == 0):
            filename = "frame%d.jpg" % count; count+=1
            cv2.imwrite(filename, frame)
    video_capture.release()
    return "Done"
    

def create_model(path):
    model = smplx.body_models.create(path, "smplx")
    return model


def plot_model(model, plot_joints=False):

    vertices = model.vertices.detach().cpu().numpy().squeeze()
    joints = model.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(vertices, model.faces,
                               vertex_colors=vertex_colors)

    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    scene = pyrender.Scene()
    scene.add(mesh)

    if plot_joints:
        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(joints_pcl)

    pyrender.Viewer(scene, use_raymond_lighting=True)
