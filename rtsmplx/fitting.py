import torch
import pytorch3d
import torch.nn as nn
import torch.optim as optim
import rtsmplx.dataset as dataset
import rtsmplx.body_model as bm
import rtsmplx.camera as cam
import pytorch3d


def opt_step(data, body_model, body=False, face=False, hands=False, lr=lr):
    image = data[0]
    image_landmarks = data[1]
    pose_image_landmarks = image_landmarks.body_landmarks()
    face_image_landmarks = image_landmarks.face_landmarks()[:17, :]
    camera = cam.Camera()

    if body == True:
        betas = body_model.body_pose
        print(betas)
        betas.requires_grad = True
        joints = body_model.get_joints()
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

    opt = optimizer(lr=lr)
    loss_pred = loss(
            pose_loss=pose_loss_pred, face_loss=face_loss_pred, hands_loss=hands_loss_pred
            )
    loss_pred.backward()
    print(betas)
    return predictions


def pose_loss(joint_coords_2d, landmarks_2d):
    return nn.MSELoss(bary_coords_2d, landmarks_2d)


def face_loss(bary_coords_2d, landmarks_2d):
    return nn.MSELoss(bary_coords_2d, landmarks_2d)

def loss(pose_loss=0, face_loss=0, hands_loss=0)
    loss_val = pose_loss + face_loss + hands_loss
    return loss_val


def optimizer(lr=1e-3):
    return optim.LBFGS(lr=lr)


def transform_bary_coords(bary_coords, bary_vertices):
    transf_bary_coords = torch.einsum("ijk,ij->ik", bary_vertices, bary_coords)
    return transf_bary_coords



if __name__ == "__main__":
    pass

