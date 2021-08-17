import torch
import pytorch3d
import torch.nn as nn
import torch.optim as optim
import rtsmplx.dataset as dataset
import rtsmplx.body_model as bm
import rtsmplx.camera as cam
import rtsmplx.lm_joint_mapping
import pytorch3d.structures
import trimesh
from torch.utils.tensorboard import SummaryWriter


def opt_step(
    image_landmarks,
    body_model,
    body_params=None,
    lr=1e-3,
    body=False,
    face=False,
    hands=False,
    writer=None,
    idx=0,
):
    pose_mapping = rtsmplx.lm_joint_mapping.get_lm_mapping()
    pose_image_landmarks = image_landmarks.body_landmarks()[pose_mapping[:, 1]]
    face_image_landmarks = image_landmarks.face_landmarks()[:17, :]
    camera = cam.Camera()
    # opt = optimizer(params, lr=lr)

    if body == True:
        if body_params == None:
            body_pose_params = body_model.body_pose
        else:
            body_pose_params = body_params
        body_pose_params.requires_grad = True
        joints = body_model.get_joints(body_pose=body_pose_params)
        joints = joints[pose_mapping[:, 0]]
        pose_prediction = camera.orthographic_projection(joints)
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
        face_predictions = camera.orthographic_projection(transf_bary_coords)
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

    # opt.zero_grad()
    loss_pred = loss(
        pose_loss=pose_loss_pred, face_loss=face_loss_pred, hands_loss=hands_loss_pred
    )
    loss_pred.backward()
    body_pose_params = body_pose_params - lr * body_pose_params.grad
    return (body_model, body_pose_params.detach())


def opt_loop(data, body_model, num_runs, body=False, face=False, hands=False, lr=1e-3):
    writer = SummaryWriter()
    image = data[0]
    landmarks = data[1]
    for i in range(num_runs):
        if (i % 5 == 0):
            print(i)
        if i > 0:
            body_model, body_pose_params = opt_step(
                landmarks,
                body_model,
                body=body,
                face=face,
                hands=hands,
                body_params=body_pose_params,
                lr=lr,
                writer=writer,
                idx=i,
            )
        else:
            body_model, body_pose_params = opt_step(
                landmarks,
                body_model,
                body=body,
                face=face,
                hands=hands,
                body_params=None,
                lr=lr,
                writer=writer,
                idx=i,
            )

    writer.close()

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


"""
def optimizer(lr=1e-3):
    return optim.LBFGS(lr=lr)
"""


def transform_bary_coords(bary_coords, bary_vertices):
    transf_bary_coords = torch.einsum("ijk,ij->ik", bary_vertices, bary_coords)
    return transf_bary_coords


if __name__ == "__main__":
    pass
