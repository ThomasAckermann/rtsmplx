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


def test_model():
    support_dir = "./../support_data/models"
    ground_truth_dir = "./../../evaluation_data/models"
    expr_dir = osp.join(support_dir, "vposer_v2_05")
    Dataset = data.EvaluationDataset(
        "./../../evaluation_data/images",
        ground_truth_dir,
        head=False,
        hands=False,
    )
    model = bm.BodyModel("./../support_data/models/smplx/", gender="male")
    vposer_model, ps = load_model(
        expr_dir,
        model_code=VPoser,
        remove_words_in_model_weights="vp_model.",
        disable_grad=True,
    )
    joints = model.get_joints()
    cam_type = "orthographic"
    vertex_distances = torch.zeros(len(Dataset))
    for i in range(len(Dataset)):
        body_model, body_pose, ocam = rtsmplx.fitting.opt_loop(
            Dataset[i],
            model,
            vposer_model,
            200,
            face=False,
            hands=False,
            lr=1e-1,
            cam_type=cam_type,
            print_every=50,
        )
        mesh = rtsmplx.fitting.get_mesh(body_model, body_pose)
        ground_truth_mesh = Dataset[i][3][0]
        vertex_distances[i] = vertex_diff(mesh, ground_truth_mesh)

    mean_v2v = torch.mean(vertex_distances)

    return mean_v2v


def vertex_diff(mesh_prediction, mesh_test):
    vertex_diff = mesh_prediction - mesh_test
    return vertex_diff


def joint_diff(joint_prediction, joint_test):
    joint_diff = joint_prediction - joint_test
    return joint_diff


if __name__ == "__main__":
    test_model()
