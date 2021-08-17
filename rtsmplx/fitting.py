import torch
import pytorch3d
import torch.nn as nn
import torch.optim as optim
import rtsmplx.dataset as dataset
import rtsmplx.body_model as bm
import rtsmplx.camera as cam
import pytorch3d


def opt_step(data, body_model, body=False, face=False, hands=False, lr=1e-3):
    image = data[0]
    image_landmarks = data[1]
    pose_image_landmarks = image_landmarks.body_landmarks()
    face_image_landmarks = image_landmarks.face_landmarks()[:17, :]
    camera = cam.Camera()
    # opt = optimizer(params, lr=lr)

    if body == True:
        body_pose_params = body_model.body_pose
        body_pose_params.requires_grad = True
        joints = body_model.get_joints(body_pose=body_pose_params)
        joints = joints[:33, :]
        pose_prediction = camera.orthographic_projection(joints)
        pose_loss_pred = pose_loss(pose_prediction, pose_image_landmarks)
    else:
        pose_loss_pred = 0

    if face == True:
        bary_coords = body_model.bary_coords
        bary_coords.requires_grad = True
        bary_vertices = body_model.bary_vertices
        transf_bary_coords = transform_bary_coords(bary_coords, bary_vertices)
        face_predictions = camera.orthographic_projection(transf_bary_coords)
        face_loss_pred = face_loss(face_predictions, face_image_landmarks)
    else:
        face_loss_pred = 0

    if hands == True:
        # Work in progress
        hands_loss_pred = 0
    else:
        hands_loss_pred = 0

    # opt.zero_grad()
    loss_pred = loss(
        pose_loss=pose_loss_pred, face_loss=face_loss_pred, hands_loss=hands_loss_pred
    )
    loss_pred.backward()
    body_pose_params = body_pose_params - lr * body_pose_params.grad
    return body_pose_params


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
