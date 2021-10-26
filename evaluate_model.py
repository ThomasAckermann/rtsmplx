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
image_dir = "./../evaluation_data/images/"
ground_truth_dir = "./../evaluation_data/ground_truth/"
cam_type = "orthographic"

if __name__ == "__main__":
    Dataset = data.EvaluationDataset(image_dir, ground_truth_dir, head=False, hands=False)
    model = bm.BodyModel("./support_data/models/smplx/", gender="male")
    vposer_model, ps = load_model(
        expr_dir, model_code=VPoser,
        remove_words_in_model_weights='vp_model.',
        disable_grad=True
    )


    fig = plt.figure()
    for index in range(len(Dataset)):
        IMG_H, IMG_W, channels = Dataset[index][0].shape
        body_model, body_pose, ocam = rtsmplx.fitting.opt_loop(Dataset[index], model, vposer_model, 100, face=False, hands=False, lr=1e-1, cam_type=cam_type, print_every=50)
        mesh = rtsmplx.fitting.get_mesh(body_model, body_pose)
        color = rtsmplx.mesh_viewer.render_trimesh_orthographic_torch(mesh, ocam, image_size=(IMG_H, IMG_W)).cpu().numpy()
        ax1 = fig.add_subplot(1,2,2)
        ax1.imshow(Dataset[index][0].detach().cpu())
        ax2 = fig.add_subplot(1,2,1)
        ax2.imshow(color[0])
        fig.savefig("./pred/{}.png".format(i))
        fig.clf()
        # diff = body_model.get_joints()[]
