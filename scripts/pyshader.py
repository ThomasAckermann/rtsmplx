import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt

tm = trimesh.load("../assets/cow/cow.obj")
m = pyrender.Mesh.from_trimesh(tm)#
scene = pyrender.Scene()
scene.add(m)
camera = pyrender.PerspectiveCamera(yfov=3, aspectRatio=1.0)
s = np.sqrt(2)/2
camera_pose = np.array([
    [0.0, -s,   s,   0.3],
    [1.0,  0.0, 0.0, 0.0],
    [0.0,  s,   s,   0.35],
    [0.0,  0.0, 0.0, 1.0],
    ])
scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0)
scene.add(light, pose=camera_pose)
r = pyrender.OffscreenRenderer(400,400)
color, depth = r.render(scene)
plt.imshow(color)
plt.show()