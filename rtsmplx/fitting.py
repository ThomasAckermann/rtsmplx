import torch
import pytorch3d
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rtsmplx.dataset as dataset
import rtsmplx.body_model as bm
import rtsmplx.camera as cam
import rtsmplx.utils as utils
import rtsmplx.lm_joint_mapping
import pytorch3d.structures
import trimesh
from torch.utils.tensorboard import SummaryWriter

import os
import human_body_prior as hbp


SUPPORT_DIRECTORY = "../support_data"
VPOSER_DIRECTORY = os.path.join(
    SUPPORT_DIRECTORY, "vposer_v2_05"
)  #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
BODY_MODEL_PATH = os.path.join(
    SUPPORT_DIRECTORY, "models/smplx/SMPLX_MALE.npz"
)  #'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads


def opt_step(
    image_landmarks,
    pose_image_landmarks,
    face_image_landmarks,
    body_model,
    opt,
    ocam,
    vposer=None,
    body_params=None,
    lr=1e-3,
    regu=1e-3,
    body=False,
    face=False,
    hands=False,
    writer=None,
    idx=0,
):
    pose_mapping = rtsmplx.lm_joint_mapping.get_lm_mapping()
    """
    pose_image_landmarks = torch.cat(
        (pose_image_landmarks, face_image_landmarks), dim=0
    )
    """

    def closure():
        if body == True:
            # joints = body_model.get_joints(body_pose=body_pose_params)
            joints = body_model.get_joints(body_pose=body_model.body_pose)
            joints = joints[pose_mapping[:, 0]]
            joints_3d = joints
            pose_prediction = ocam.forward(joints)
            pose_loss_pred = pose_loss(pose_prediction, pose_image_landmarks)
            if writer != None:
                writer.add_scalar("Pose Loss", pose_loss_pred.detach(), idx)
        else:
            pose_loss_pred = 0

        if face == True:
            bary_coords = body_model.bary_coords
            bary_coords.requires_grad = True
            bary_vertices = body_model.bary_vertices
            transf_bary_coords = transform_bary_coords(bary_coords, bary_vertices)
            face_predictions = ocam.forward(transf_bary_coords)
            face_loss_pred = face_loss(face_predictions, face_image_landmarks)
            if writer != None:
                writer.add_scalar("Face Loss", face_loss_pred.detach(), idx)
        else:
            face_loss_pred = 0

        if hands == True:
            # Work in progress
            hands_loss_pred = 0
            if writer != None:
                writer.add_scalar("Hands Loss", hands_loss_pred.detach(), idx)
        else:
            hands_loss_pred = 0
        body_pose_param = body_model.body_pose

        # VPoser prior
        vposer_prior = None
        if vposer:
            vposer_joint_rot = (
                vposer.forward(body_model.body_pose)["pose_body"]
                .reshape((-1, 63))
                .detach()
            )
            vposer_joints = body_model.get_joints(body_pose=vposer_joint_rot)[
                pose_mapping[:, 0]
            ].detach()
            vposer_prior = vposer_loss(joints_3d, vposer_joints)
            if writer != None:
                writer.add_scalar("VPoser Prior", vposer_prior.detach(), idx)

        body_pose_prior = torch.linalg.norm(body_model.body_pose)
        if writer != None:
            writer.add_scalar("Joint Rot Prior", body_pose_prior.detach(), idx)

        opt.zero_grad()
        loss_pred = loss(
            pose_loss_pred,
            face_loss_pred,
            hands_loss_pred,
            body_pose_param,
            body_pose_prior=body_pose_prior,
            vposer_prior=vposer_prior,
            regu=regu,
        )
        if writer != None:
            writer.add_scalar("Loss with Regularization", loss_pred.detach(), idx)
        loss_pred.backward()
        return loss_pred

    opt.step(closure)
    return (body_model, ocam)


def run(
    num_runs,
    landmarks,
    pose_image_landmarks,
    face_image_landmarks,
    body_model,
    opt,
    ocam,
    body=False,
    face=False,
    hands=False,
    body_params=None,
    lr=1e-3,
    regularization=1e-3,
    print_every=50,
    vposer=None,
):
    writer = SummaryWriter()
    for i in range(1, num_runs):
        # if i % print_every == 0:
        # print(i)

        body_model, ocam = opt_step(
            landmarks,
            pose_image_landmarks,
            face_image_landmarks,
            body_model,
            opt,
            ocam,
            body=body,
            face=face,
            hands=hands,
            body_params=body_params,
            lr=lr,
            regu=regularization,
            writer=writer,
            vposer=vposer,
            idx=i,
        )
        writer.close()
    return body_model, ocam


def opt_loop(
    data,
    body_model,
    num_runs,
    body=False,
    face=False,
    hands=False,
    lr=1e-3,
    regularization=1e-2,
    vposer=None,
):
    image = data[0]
    landmarks = data[1]
    pose_mapping = rtsmplx.lm_joint_mapping.get_lm_mapping()
    pose_image_landmarks = landmarks.body_landmarks()[pose_mapping[:, 1]]
    # face_image_landmarks = landmarks.face_landmarks()[17:, :]
    face_image_landmarks = None
    ocam = cam.OrthographicCamera()
    itern = 0
    for param in body_model.parameters():
        if itern == 2:
            param.requires_grad = True
        else:
            param.requires_grad = False
        itern += 1
    # opt = optimizer(body_model.parameters(), lr=lr)
    opt = optimizer(ocam.parameters(), lr=lr)
    # opt = optimizer(list(body_model.parameters()) + list(ocam.parameters()), lr=lr)
    body_model, ocam = run(
        int(num_runs / 2),
        landmarks,
        pose_image_landmarks,
        face_image_landmarks,
        body_model,
        opt,
        ocam,
        body=body,
        face=face,
        hands=hands,
        body_params=None,
        lr=lr,
        regularization=regularization,
        vposer=vposer,
    )
    # opt = optimizer(ocam.parameters(), lr=lr)
    opt = optimizer(body_model.parameters(), lr=lr)
    body_model, ocam = run(
        int(num_runs / 2),
        landmarks,
        pose_image_landmarks,
        face_image_landmarks,
        body_model,
        opt,
        ocam,
        body=body,
        face=face,
        hands=hands,
        body_params=None,
        lr=lr,
        regularization=regularization,
        vposer=vposer,
    )
    """
    lr = lr * 0.1
    opt = optimizer(list(body_model.parameters()) + list(ocam.parameters()), lr=lr)
    body_model, ocam = run(
        num_runs,
        landmarks,
        pose_image_landmarks,
        face_image_landmarks,
        body_model,
        opt,
        ocam,
        body=body,
        face=face,
        hands=hands,
        body_params=None,
        lr=lr,
        regularization=regularization,
        vposer=vposer,
    )
    """
    body_pose_params = body_model.body_pose

    return body_model, body_pose_params, ocam


def get_mesh(body_model, body_pose):
    faces = body_model.faces.reshape(-1, 3).detach().numpy()
    vertices = (
        body_model.forward(body_pose=body_pose).vertices.reshape(-1, 3).detach().numpy()
    )
    tri_mesh = trimesh.base.Trimesh(vertices=vertices, faces=faces)
    return tri_mesh


def pose_loss(joint_coords_2d, landmarks_2d):
    loss_func = mse_loss()
    return loss_func(joint_coords_2d, landmarks_2d)


def face_loss(bary_coords_2d, landmarks_2d):
    loss_func = mse_loss()
    return loss_func(bary_coords_2d, landmarks_2d)


def vposer_loss(joints_3d, vposer_joints_3d):
    loss_func = mse_loss()
    return loss_func(joints_3d, vposer_joints_3d)


def loss(
    pose_loss,
    face_loss,
    hands_loss,
    body_pose,
    body_pose_prior=None,
    vposer_prior=None,
    regu=1e-3,
):
    loss_val = pose_loss + face_loss + hands_loss

    # priors
    if body_pose_prior:
        loss_val = loss_val + regu * body_pose_prior
    if vposer_prior:
        loss_val = loss_val + regu * vposer_prior

    return loss_val


def mse_loss():
    return nn.MSELoss()


def optimizer(params, lr=1e-3):
    return optim.Adam(params, lr=lr)


def transform_bary_coords(bary_coords, bary_vertices):
    transf_bary_coords = torch.einsum("ijk,ij->ik", bary_vertices, bary_coords)
    return transf_bary_coords


def plot_landmarks(ocam, body_model, image_landmarks):
    pose_mapping = rtsmplx.lm_joint_mapping.get_lm_mapping()
    pose_image_landmarks = image_landmarks.body_landmarks()
    """
    face_image_landmarks = image_landmarks.face_landmarks()[17:, :]
    pose_image_landmarks = torch.cat(
        (pose_image_landmarks, face_image_landmarks), dim=0
    )
    """
    pose_image_landmarks = pose_image_landmarks[pose_mapping[:, 1]]

    joints = body_model.get_joints(body_pose=body_model.body_pose)
    joints = joints[pose_mapping[:, 0]]
    pose_prediction = ocam.forward(joints)
    return (pose_image_landmarks.numpy(), pose_prediction.detach().numpy())


if __name__ == "__main__":
    pass
