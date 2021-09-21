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
        )
BODY_MODEL_PATH = os.path.join(
        SUPPORT_DIRECTORY, "models/smplx/SMPLX_MALE.npz"
        )

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
        device="cpu",
        print_every=200,
        ):
    pose_mapping = rtsmplx.lm_joint_mapping.get_lm_mapping()
    """
    pose_image_landmarks = torch.cat(
        (pose_image_landmarks, face_image_landmarks), dim=0
    )
    """

    def closure():
        if body == True:
            # print(vposer.decode(body_model.latent_j))
            body_model.body_pose = nn.Parameter(vposer.decode(body_model.latent_j)["pose_body"])
            body_pose = vposer.decode(body_model.latent_j)["pose_body"]
            # body_model.body_pose.requires_grad = False
            joints = body_model.get_joints(body_pose=body_pose)#body_model.body_pose)
            joints = joints[pose_mapping[:, 0]]
            joints_3d = joints
            pose_prediction = ocam.forward(joints).to(device=device)
            pose_loss_pred = pose_loss(pose_prediction, pose_image_landmarks).to(device=device)
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
        body_pose_param = body_pose #body_model.body_pose


        # body_pose_prior = torch.linalg.norm(body_model.body_pose).to(device=device)
        body_pose_prior = torch.linalg.norm(body_pose).to(device=device)
        if writer != None:
            writer.add_scalar("Joint Rot Prior", body_pose_prior.detach(), idx)

        opt.zero_grad()
        loss_pred = loss(
                pose_loss_pred,
                face_loss_pred,
                hands_loss_pred,
                body_pose_param,
                body_pose_prior=body_pose_prior,
                regu=regu,
                )
        if writer != None:
            writer.add_scalar("Loss with Regularization", loss_pred.detach(), idx)
        loss_pred.backward()
        if idx % print_every == 0:
            print("Iteration:", idx, "Loss:", loss_pred.detach().cpu().numpy())
        body_model.body_pose = nn.Parameter(body_pose)
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
        print_every=200,
        vposer=None,
        ):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    writer = SummaryWriter()
    for i in range(1, num_runs):
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
                device=device,
                print_every=print_every,
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
        cam_type="perspective",
        print_every=200
        ):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    image = data[0]
    landmarks = data[1]
    pose_mapping = rtsmplx.lm_joint_mapping.get_lm_mapping()
    pose_image_landmarks = landmarks.body_landmarks()[pose_mapping[:, 1]].to(device=device)
    # face_image_landmarks = landmarks.face_landmarks()[17:, :]
    face_image_landmarks = None
    if cam_type == "perspective":
        ocam = cam.PerspectiveCamera().to(device=device)
    else:
        ocam = cam.OrthographicCamera().to(device=device)
    itern = 0
    for param in body_model.parameters():
        if itern == 10:
            param.requires_grad = True
        else:
            param.requires_grad = False
        itern += 1

    print("Start Optimizing Camera")
    opt = optimizer(ocam.parameters(), lr=lr)
    body_model, ocam = run(
            int(num_runs / 4),
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
            print_every=print_every,
            )
    opt = optimizer(body_model.parameters(), lr=lr)
    print("Start Optimizing Body Pose")
    body_model, ocam = run(
            int(num_runs / 4),
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
            print_every=print_every,
            )
    print("Start Optimizing Camera")
    opt = optimizer(ocam.parameters(), lr=lr)
    body_model, ocam = run(
            int(num_runs / 4),
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
            print_every=print_every,
            )
    opt = optimizer(body_model.parameters(), lr=lr)
    print("Start Optimizing Body Pose")
    body_model, ocam = run(
            int(num_runs / 4),
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
            print_every=print_every,
            )

    """
    opt = optimizer(list(body_model.parameters()) + list(ocam.parameters()), lr=lr)
    print("Start Optimizing Camera and Pose together")
    body_model, ocam = run(
            int(num_runs / 3),
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
            print_every=print_every,
            )
    """

    body_pose_params = body_model.body_pose
    return body_model, body_pose_params, ocam


def get_mesh(body_model, body_pose):
    faces = body_model.faces.reshape(-1, 3).detach().cpu().numpy()
    vertices = (
            body_model.forward(body_pose=body_pose).vertices.reshape(-1, 3).detach().cpu().numpy()
            )
    tri_mesh = trimesh.base.Trimesh(vertices=vertices, faces=faces)
    return tri_mesh


def pose_loss(joint_coords_2d, landmarks_2d):
    loss_func = mse_loss()
    return loss_func(joint_coords_2d, landmarks_2d)


def face_loss(bary_coords_2d, landmarks_2d):
    loss_func = mse_loss()
    return loss_func(bary_coords_2d, landmarks_2d)



def loss(
        pose_loss,
        face_loss,
        hands_loss,
        body_pose,
        body_pose_prior=None,
        regu=1e-4,
        ):
    loss_val = pose_loss + face_loss + hands_loss

    # priors
    if body_pose_prior:
        loss_val = loss_val + regu * body_pose_prior

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
    return (pose_image_landmarks.detach().cpu().numpy(), pose_prediction.detach().cpu().numpy())


if __name__ == "__main__":
    pass
