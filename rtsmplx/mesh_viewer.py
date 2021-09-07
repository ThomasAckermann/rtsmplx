import trimesh
import pyrender
import matplotlib.pyplot as plt
import rtsmplx.camera as smplxcam
import numpy as np
import torch


def render_trimesh(tri_mesh):
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)


def render_trimesh_orthographic(
    tri_mesh, ocam, xmag=1, ymag=1.0, intensity=1, imgh=400, imgw=400, offset=1
):
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3, 1.0])
    scene.add(mesh)
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)
    camera_pose = ocam.get_cam_transform().detach().numpy()
    # camera_pose[2][2] = -1 * camera_pose[2][2]
    camera_pose[:3, 3] = np.array([0.0, 0.0, 0.0])
    camera_pose[2, 3] = camera_pose[2, 3] + offset
    scene.add(camera, pose=camera_pose)
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(imgh, imgw)
    color, depth = r.render(scene)
    return color, depth


def render_trimesh_no_transform(trimesh, xmag=1.0, ymag=1.0, imgh=400, imgw=400):
    mesh = pyrender.Mesh.from_trimesh(trimesh)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3, 1.0])
    scene.add(mesh)
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
    scene.add(light)

    r = pyrender.OffscreenRenderer(imgh, imgw)
    color, depth = r.render(scene)
    return color, depth
