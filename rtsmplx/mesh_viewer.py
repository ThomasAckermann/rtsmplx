import pyrender
import trimesh
import numpy as np


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
        self.mesh_constructir = trimesh.Trimesh
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
        if not self.viewer.is_active
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
    def __init__():

    def render(mesh)

