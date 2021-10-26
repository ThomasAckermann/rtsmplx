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

support_dir = './../support_data/models'
expr_dir = osp.join(support_dir,'vposer_v2_05')
body_model_dir = osp.join(support_dir,'smplx/')


if __name__ == "__main__":
    Dataset = data.ImageDataset("./../data/tiktok/", head=False, hands=False, silhouette_dir="./data/tiktok_silhouette/")
    model = bm.BodyModel(body_model_dir, gender="male")
    vposer_model, ps = load_model(expr_dir, model_code=VPoser,
                                remove_words_in_model_weights='vp_model.',
                                disable_grad=True)
    joints = model.get_joints()
    IMAGE_ID = 0
    IMG_H = Dataset[IMAGE_ID][1].image_height
    IMG_W = Dataset[IMAGE_ID][1].image_width
    cam_type = "orthographic"


    fig = plt.figure()
    for i in range(20):#range(len(Dataset)):
        if i == 0:
            body_model, body_pose, new_cam = rtsmplx.fitting.video_opt_loop(
                    Dataset[IMAGE_ID],
                    model, vposer_model,
                    100,
                    face=False,
                    hands=False,
                    lr=1e-1,
                    cam_type=cam_type,
                    print_every=50
                    )
            ocam = new_cam
        body_model, body_pose, new_cam = rtsmplx.fitting.video_opt_loop(
                Dataset[IMAGE_ID],
                model,
                vposer_model,
                100,
                previous_model=body_model,
                previous_cam=new_cam,
                face=False,
                hands=False,
                lr=1e-1,
                cam_type=cam_type,
                print_every=50
                )
        mesh = rtsmplx.fitting.get_mesh(body_model, body_pose)
        # color = rtsmplx.mesh_viewer.render_trimesh_orthographic_torch(mesh, new_cam, image_size=(IMG_H, IMG_W)).cpu().numpy()
        color = rtsmplx.mesh_viewer.render_silhouette_orthographic(mesh, new_cam, image_size=(IMG_H, IMG_W)).cpu().numpy()
        # plt.imshow(color[0])
        # plt.savefig("pred/{}.png".format(i))
        ax1 = fig.add_subplot(1,2,2)
        ax1.imshow(Dataset[i][3].detach().cpu())
        ax2 = fig.add_subplot(1,2,1)
        ax2.imshow(color[0])
        fig.savefig("pred/{}.png".format(i))
        # plt.show()
        fig.clf()

