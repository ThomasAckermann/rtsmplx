import rtsmplx.dataset as data
import rtsmplx.body_model as bm
import rtsmplx.landmarks as lm
import rtsmplx.fitting
import rtsmplx.mesh_viewer

import torch
import numpy as np
import matplotlib.pyplot as plt
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

from os import path as osp

support_dir = './support_data/models'
expr_dir = osp.join(support_dir,'vposer_v2_05')


if __name__ == "__main__":

    Dataset = data.ImageDataset("./data/tiktok/", head=False, hands=False)
    model = bm.BodyModel("./support_data/models/smplx/", gender="male")
    vposer_model, ps = load_model(expr_dir, model_code=VPoser,
            remove_words_in_model_weights='vp_model.',
            disable_grad=True)
    joints = model.get_joints()
    IMAGE_ID = 0
    IMG_H = Dataset[IMAGE_ID][1].image_height
    IMG_W = Dataset[IMAGE_ID][1].image_width
    cam_type = "orthographic"

    for i in range(len(Dataset)):
        ipachinko f i == 0:
            body_model, body_pose, new_cam = rtsmplx.fitting.opt_loop(Dataset[IMAGE_ID], model, vposer_model, 1000, face=False, hands=False, lr=5e-1, cam_type=cam_type, print_every=50)
            ocam = new_cam
        body_model, body_pose, new_cam = rtsmplx.fitting.opt_loop(Dataset[IMAGE_ID], model, vposer_model, 100, face=False, hands=False, lr=5e-1, cam_type=cam_type, print_every=50)
        mesh = rtsmplx.fitting.get_mesh(body_model, body_pose)
        points = ocam.forward(torch.Tensor(mesh.vertices).to(device="cuda")).detach().cpu().numpy()
        plt.scatter(points[:,0], -1*points[:,1])
        plt.savefig("pred/{}.png".format(i))
        plt.clf()
