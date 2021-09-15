import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["LIBGL_ALWAYS_INDIRECT"] = "0"

import rtsmplx.dataset as data
import rtsmplx.body_model as bm
import rtsmplx.landmarks as lm
import rtsmplx.fitting
import rtsmplx.mesh_viewer

import torch
import numpy as np
import matplotlib.pyplot as plt
# from human_body_prior.body_model.body_model import BodyModel
# from human_body_prior.tools.model_loader import load_model
# from human_body_prior.models.vposer_model import VPoser

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":

    support_dir = './support_data'
    expr_dir = os.path.join(support_dir,'vposer_v2_05')
    Dataset = data.ImageDataset("./data/", head=False, hands=True)
    model = bm.BodyModel("./support_data/models/smplx/", gender="male").to(device=device)
    # vposer_model, ps = load_model(expr_dir, model_code=VPoser,
    #                               remove_words_in_model_weights='vp_model.',
    #                               disable_grad=True)
    joints = model.get_joints()

    IMAGE_ID =4 #1 #4
    IMG_H = Dataset[IMAGE_ID][1].image_height
    IMG_W = Dataset[IMAGE_ID][1].image_width
    cam_type = "orthographic" # "perspective"
    body_model, body_pose, ocam = rtsmplx.fitting.opt_loop(Dataset[IMAGE_ID], model, 600, body=True, face=False, hands=False, lr=1e-1, regularization=2e-4, vposer=None, cam_type=cam_type, print_every=30)
    image_lm, model_lm = rtsmplx.fitting.plot_landmarks(ocam, body_model, Dataset[IMAGE_ID][1].detach().cpu().numpy()
