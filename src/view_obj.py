import trimesh
import pyrender
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Path to .obj file.")

arguments = parser.parse_args()


def view_obj(path):
    trimesh_data = trimesh.load(path)
    mesh = pyrender.Mesh.from_trimesh(trimesh_data)
    scene = pyrender.Scene()
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    view_obj(arguments.path)
