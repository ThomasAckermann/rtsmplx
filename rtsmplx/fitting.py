import torch
import pytorch3d
import torch.nn as nn
import torch.optim as optim
import rtsmplx.dataset as dataset
import rtsmplx.body_model as bm
import rtsmplx.camera as cam
import pytorch3d


def forward(data, body_model):
    image = data[0]
    image_landmarks = data[1]
    image_landmarks = torch.flip(image_landmarks.face_landmarks())
    bary_coords = body_model.bary_coords
    bary_vertices = body_model.bary_vertices
    transf_bary_coords = transform_bary_coords(bary_coords, bary_vertices)
    camera = cam.Camera()
    predictions = camera.orthographic_projection(transf_bary_coords)
    return predictions


def backward(data, bary_coords_2d):
    landmarks_2d = data[1].body_lm
    pred_loss = loss(landmarks_2d, bary_coords_2d)
    return None


def loss(bary_coords_2d, landmarks_2d):
    return nn.MSELoss(bary_coords_2d, landmarks_2d)


def optimizer(lr=1e-3):
    return optim.LBFGS(lr=lr)


def transform_bary_coords(bary_coords, bary_vertices):
    transf_bary_coords = torch.einsum("ijk,ij->ik", bary_vertices, bary_coords)
    return transf_bary_coords



if __name__ == "__main__":
    pass

