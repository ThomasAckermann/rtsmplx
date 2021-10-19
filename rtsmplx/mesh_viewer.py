import trimesh
import pyrender
import matplotlib.pyplot as plt
import rtsmplx.camera as smplxcam
import rtsmplx.utils as utils
import numpy as np
import torch
import pytorch3d

from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        PointLights,
        DirectionalLights,
        Materials,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        SoftSilhouetteShader,
        TexturesUV,
        TexturesVertex
        )


def render_trimesh(tri_mesh):
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)


def render_trimesh_orthographic(
        tri_mesh, ocam, xmag=1, ymag=1.0, intensity=1, imgh=400, imgw=400, offset=1
        ):
    device="cpu"
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3, 1.0])
    scene.add(mesh)
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)
    camera_pose = ocam.get_cam_transform(cpu=True).detach().cpu().numpy()
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


def render_trimesh_perspective_torch(trimesh, ocam, image_size=512):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mesh = utils.trimesh_to_torch(trimesh).to(device=device)
    T, R = utils.get_torch_trans_format(ocam.translation.detach(), ocam.rotation.detach())
    fx = ocam.focal_length_x
    fy = ocam.focal_length_y
    center = ocam.center
    calibration_mat = torch.tensor([[
        [fx, 0, center[0], 0],
        [0, fy, center[1], 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
        ]]).to(device=device)

    render_camera = pytorch3d.renderer.cameras.FoVPerspectiveCameras(R=R, T=T, K=calibration_mat, device=device)
    raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            )

    lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=render_camera,
                raster_settings=raster_settings
                ),
            shader=SoftPhongShader(
                device=device,
                cameras=render_camera,
                lights=lights
                )
            )

    image = renderer(mesh)
    return image


def render_trimesh_orthographic_torch(trimesh, ocam, image_size=512, zfar=100, znear=1.0, max_y=1.0, min_y=-1.0, max_x=1.0, min_x=-1.0,):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mesh = utils.trimesh_to_torch(trimesh).to(device=device)
    T, R = utils.get_torch_trans_format(ocam.translation.detach(), ocam.rotation.detach())
    T.requires_grad = False
    R.requires_grad = False
    scale = ocam.scale.reshape((1,3)).detach()
    frustum_max = ocam.frustum_max.detach()
    frustum_min = ocam.frustum_min.detach()


    render_camera = pytorch3d.renderer.cameras.FoVOrthographicCameras(
            R=R,
            T=T,
            device=device,
            zfar=frustum_max[0],
            znear=frustum_min[0],
            max_y=frustum_max[1],
            min_y=frustum_min[1],
            max_x=frustum_max[2],
            min_x=frustum_min[2],
            )
    raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            )

    lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=render_camera,
                raster_settings=raster_settings
                ),
            shader=SoftPhongShader(
                device=device,
                cameras=render_camera,
                lights=lights
                )
            )

    image = renderer(mesh)
    return image


def render_silhouette_orthographic(trimesh, ocam, image_size=512, zfar=100, znear=1.0, max_y=1.0, min_y=-1.0, max_x=1.0, min_x=-1.0,):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mesh = utils.trimesh_to_torch(trimesh).to(device=device)
    T, R = utils.get_torch_trans_format(ocam.translation.detach(), ocam.rotation.detach())
    scale = ocam.scale.detach()
    sigma = 1e-4
    frustum_max = ocam.frustum_max.detach()
    frustum_min = ocam.frustum_min.detach()


    render_camera = pytorch3d.renderer.cameras.FoVOrthographicCameras(
            R=R,
            T=T,
            device=device,
            zfar=frustum_max[0],
            znear=frustum_min[0],
            max_y=frustum_max[1],
            min_y=frustum_min[1],
            max_x=frustum_max[2],
            min_x=frustum_min[2],
            )

    raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 -1. )*sigma,
            faces_per_pixel=1,
            )

    renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=render_camera,
                raster_settings=raster_settings
                ),
            shader=SoftSilhouetteShader()
            )

    image = renderer(mesh)
    return image
