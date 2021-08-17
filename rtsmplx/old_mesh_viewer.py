import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import torch

import pytorch3d
from pytorch3d.structures import Meshes
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
    TexturesUV,
    TexturesVertex
)

class MeshViewer:
    def __init__(
            self,
            width=1200,
            height=800,
            body_color=(1.0, 1.0, 0.9, 1.0),
            registered_keys=None,
            ):
        super(MeshViewer, self).__init__()
        self.material_constructor = pyreder.MetallicRoughnessMaterial
        self.mesh_constructor = trimesh.Trimesh
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh
        self.transform = trimesh.transformation.rotation_matrix

        self.body_color = body_color
        self.scene = pyrender.Scene(
                bg_color=[0.0, 0.0, 0.0, 1.0], ambient_light=(0.3, 0.3, 0.3)
                )
        camera = pyrender.PerspectiveCamera(
                yfov=(np.pi / 3), aspectRatio=(width / height)
                )
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 3])
        self.scene.add(camera, pose=camera_pose)
        self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, viewport_size=(width, height), cull_faces=False, run_in_thread=True, registered_keys=registered_keys)

    def is_active(self):
        return self.viewer.is_active

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()


    def create_mesh(self, vertices, faces, color=(0.3, 0.3, 0.3, 1.0), wireframe=False):
        material=self.material_constructor(metallicFactor=0.0, alphaMode="Blend", baseColorFactor=color)
        mesh = self.mesh_constructor(vertices, faces)
        rotation = self.transform(np.radians(180), [1,0,0])
        return self.trimesh_to_pymesh(mesh, material=material)

    def update_mesh(self, vertices, faces):
        if not self.viewer.is_active:
            return
        self.viewer.render_lock.acquire()
        for node in self.scene.get_nodes():
            if node.name == "body_mesh":
                self.scene.remove_node(node)
                break
        body_mesh = self.create_mesh(
                vertices, faces, color=self.body_color)
        self.scene.add(body_mesh, name="body_mesh")
        self.viewer.render_lock.release()


class TorchMeshViewer:
    def __init__(self, mesh):
        self.mesh = mesh

    def render(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        R, T = look_at_view_transform(2.7, 0, 180) 
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )


        images = renderer(self.mesh)
        plt.figure(figsize=(10, 10))
        plt.imshow(images[0, ..., :3].cpu().numpy())
        plt.axis("off");




