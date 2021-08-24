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


def opt_step(
    image_landmarks,
    body_model,
    opt,
    ocam,
    body_params=None,
    lr=1e-3,
    body=False,
    face=False,
    hands=False,
    writer=None,
    idx=0,
):
    pose_mapping = rtsmplx.lm_joint_mapping.get_lm_mapping()
    pose_image_landmarks = image_landmarks.body_landmarks()
    face_image_landmarks = image_landmarks.face_landmarks()[17:, :]
    """
    pose_image_landmarks = torch.cat(
        (pose_image_landmarks, face_image_landmarks), dim=0
    )
    """
    pose_image_landmarks = pose_image_landmarks[pose_mapping[:, 1]]

    if body == True:
        """
        if body_params == None:
            body_pose_params = body_model.body_pose
        else:
            body_pose_params = body_params
        body_pose_params.requires_grad = True
        """
        # joints = body_model.get_joints(body_pose=body_pose_params)
        joints = body_model.get_joints(body_pose=body_model.body_pose)
        joints = joints[pose_mapping[:, 0]]
        pose_prediction = ocam.orthographic_projection(joints)
        """
        pose_prediction = rigid_landmark_transform(
            pose_prediction, pose_image_landmarks
        )
        """
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
        face_predictions = ocam.orthographic_projection(transf_bary_coords)
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

    loss_pred = loss(
        pose_loss=pose_loss_pred, face_loss=face_loss_pred, hands_loss=hands_loss_pred
    )
    opt.zero_grad()
    loss_pred.backward()
    opt.step()
    return (body_model, ocam)


def opt_loop(data, body_model, num_runs, body=False, face=False, hands=False, lr=1e-3):
    writer = SummaryWriter()
    image = data[0]
    landmarks = data[1]
    ocam = cam.OrthographicCamera()
    opt = optimizer(body_model.parameters(), lr=lr)
    for i in range(num_runs):
        if i % 5 == 0:
            print(i)
        if i > 0:
            # body_model, body_pose_params, ocam = opt_step(
            body_model, ocam = opt_step(
                landmarks,
                body_model,
                opt,
                ocam,
                body=body,
                face=face,
                hands=hands,
                # body_params=body_pose_params,
                body_params=None,
                lr=lr,
                writer=writer,
                idx=i,
            )
        else:
            # body_model, body_pose_params, ocam = opt_step(
            body_model, ocam = opt_step(
                landmarks,
                body_model,
                opt,
                ocam,
                body=body,
                face=face,
                hands=hands,
                body_params=None,
                lr=lr,
                writer=writer,
                idx=i,
            )

            writer.close()
    body_pose_params = body_model.body_pose

    return body_model, body_pose_params


def get_mesh(body_model, body_pose):
    faces = body_model.faces.reshape(-1, 3).detach().numpy()
    vertices = (
        body_model.forward(body_pose=body_pose).vertices.reshape(-1, 3).detach().numpy()
    )
    # torch_mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces)
    tri_mesh = trimesh.base.Trimesh(vertices=vertices, faces=faces)
    return tri_mesh


def pose_loss(joint_coords_2d, landmarks_2d):
    loss_func = mse_loss()
    return loss_func(joint_coords_2d, landmarks_2d)


def face_loss(bary_coords_2d, landmarks_2d):
    return nn.MSELoss(bary_coords_2d, landmarks_2d)


def loss(pose_loss=0, face_loss=0, hands_loss=0):
    loss_val = pose_loss + face_loss + hands_loss
    return loss_val


def mse_loss():
    return nn.MSELoss()


def optimizer(params, lr=1e-3):
    return optim.Adam(params, lr=lr)


def transform_bary_coords(bary_coords, bary_vertices):
    transf_bary_coords = torch.einsum("ijk,ij->ik", bary_vertices, bary_coords)
    return transf_bary_coords


def rigid_landmark_transform(pose_landmarks, model_landmarks):
    # model_landmarks = torch.matmul(model_landmarks, utils.rot_mat_2d(torch.Tensor([np.pi])))
    body_axis_pose = (pose_landmarks[12] - pose_landmarks[0])
    body_axis_pose = body_axis_pose.detach().numpy()
    body_axis_model = (model_landmarks[12] - model_landmarks[0])
    body_axis_model = body_axis_model.detach().numpy()

    # translation
    dist_pose_model = pose_landmarks[11] - model_landmarks[11]
    model_landmarks = torch.add(model_landmarks, dist_pose_model)

    # scaling
    scaling = torch.Tensor(
        [
            [body_axis_pose[0] / body_axis_model[0], 0],
            [0, body_axis_pose[1] / body_axis_model[1]],
        ]
    )
    model_landmarks = torch.matmul(model_landmarks, scaling)

    # rotation
    rotation_angle = utils.angle_between(body_axis_pose, body_axis_model)
    model_landmarks = torch.matmul(model_landmarks, utils.rot_mat_2d(-1 * rotation_angle))

    return model_landmarks

def plot_landmarks(body_model, image_landmarks):
    ocam = cam.OrthographicCamera()
    pose_mapping = rtsmplx.lm_joint_mapping.get_lm_mapping()
    pose_image_landmarks = image_landmarks.body_landmarks()
    face_image_landmarks = image_landmarks.face_landmarks()[17:, :]
    pose_image_landmarks = torch.cat(
        (pose_image_landmarks, face_image_landmarks), dim=0
    )
    pose_image_landmarks = pose_image_landmarks[pose_mapping[:, 1]]

    joints = body_model.get_joints(body_pose=body_model.body_pose)
    joints = joints[pose_mapping[:, 0]]
    pose_prediction = ocam.orthographic_projection(joints)
    pose_prediction = rigid_landmark_transform(
        pose_prediction, pose_image_landmarks
    )
    return (pose_image_landmarks.numpy(), pose_prediction.detach().numpy())


if __name__ == "__main__":
    pass
