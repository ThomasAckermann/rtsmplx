import trimesh
import pyrender
import matplotlib.pyplot as plt


def render_trimesh(tri_mesh):
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)

