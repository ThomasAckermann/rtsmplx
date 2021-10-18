import torch
import pytorch3d
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rtsmplx.dataset as dataset
import rtsmplx.body_model as bm
import rtsmplx.camera as cam
import rtsmplx.utils as utils
import rtsmplx.loss
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
        vposer,
        model_loss,
        previous_model=None,
        lr=1e-3,
        face=False,
        hands=False,
        writer=None,
        idx=0,
        device="cpu",
        print_every=200,
        interpenetration=False,
        ):
    pose_mapping = rtsmplx.lm_joint_mapping.get_lm_mapping()
    """
    pose_image_landmarks = torch.cat(
        (pose_image_landmarks, face_image_landmarks), dim=0
    )
    """

    def closure():
        body_model.body_pose = nn.Parameter(vposer.decode(body_model.latent_j)["pose_body"])
        body_pose = vposer.decode(body_model.latent_j)["pose_body"]
        joints = body_model.get_joints(body_pose=body_pose)
        joints = joints[pose_mapping[:, 0]]
        joints_3d = joints
        pose_prediction = ocam.forward(joints).to(device=device)

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

        # Create the search tree
        """
        pen_loss = None
        search_tree = None
        pen_distance = None
        filter_faces = None
        if interpenetration == True:
            from mesh_intersection.bvh_search_tree import BVH
            import mesh_intersection.loss as collisions_loss
            from mesh_intersection.filter_faces import FilterFaces

            assert use_cuda, 'Interpenetration term can only be used with CUDA'
            assert torch.cuda.is_available(), \
                    'No CUDA Device! Interpenetration term can only be used' + \
                    ' with CUDA'

            search_tree = BVH(max_collisions=max_collisions)

            pen_distance = \
                    collisions_loss.DistanceFieldPenetrationLoss(
                    sigma=df_cone_height, point2plane=point2plane,
                    vectorized=True, penalize_outside=penalize_outside)

            if part_segm_fn:
                # Read the part segmentation
                part_segm_fn = os.path.expandvars(part_segm_fn)
                with open(part_segm_fn, 'rb') as faces_parents_file:
                    face_segm_data = pickle.load(faces_parents_file,
                                                encoding='latin1')
                faces_segm = face_segm_data['segm']
                faces_parents = face_segm_data['parents']
                # Create the module used to filter invalid collision pairs
                filter_faces = FilterFaces(
                    faces_segm=faces_segm, faces_parents=faces_parents,
                    ign_part_pairs=ign_part_pairs).to(device=device)
        """

        opt.zero_grad()
        loss = model_loss.forward(body_pose_param, pose_prediction, pose_image_landmarks, previous_model=previous_model)
        if writer != None:
            writer.add_scalar("Loss with Regularization", loss.detach(), idx)
        loss.backward()
        if idx % print_every == 0:
            print("Iteration:", idx, "Loss:", loss.detach().cpu().numpy())
        body_model.body_pose = nn.Parameter(body_pose)
        return loss

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
        vposer,
        model_loss,
        previous_model=None,
        face=False,
        hands=False,
        lr=1e-3,
        print_every=200,
        writer=None,
        idx=0,
        interpenetration=False,
        ):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for i in range(idx, idx + num_runs):
        body_model, ocam = opt_step(
                landmarks,
                pose_image_landmarks,
                face_image_landmarks,
                body_model,
                opt,
                ocam,
                vposer,
                model_loss,
                previous_model=previous_model,
                face=face,
                hands=hands,
                lr=lr,
                writer=writer,
                idx=i,
                device=device,
                print_every=print_every,
                interpenetration=interpenetration,
                )
        idx = i

    return body_model, ocam, idx


def opt_loop(
        data,
        body_model,
        vposer,
        num_runs,
        face=False,
        hands=False,
        lr=1e-3,
        cam_type="perspective",
        print_every=200,
        interpenetration=False,
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
    model_loss = rtsmplx.loss.ModelLoss()
    itern = 0
    for param in body_model.parameters():
        if itern in [2, 10]:
            param.requires_grad = True
        else:
            param.requires_grad = False
        itern += 1
    writer = SummaryWriter()

    idx = 0
    print("Start Optimizing Camera")
    opt = optimizer(list(ocam.parameters()) + list(model_loss.parameters()), lr=lr)
    body_model, ocam, idx = run(
            int(num_runs / 4),
            landmarks,
            pose_image_landmarks,
            face_image_landmarks,
            body_model,
            opt,
            ocam,
            vposer,
            model_loss,
            face=face,
            hands=hands,
            lr=lr,
            print_every=print_every,
            writer=writer,
            idx=idx,
            interpenetration=interpenetration,
            )
    print(idx)
    opt = optimizer(list(body_model.parameters()) + list(model_loss.parameters()), lr=lr)
    print("Start Optimizing Body Pose")
    body_model, ocam, idx = run(
            int(num_runs / 4),
            landmarks,
            pose_image_landmarks,
            face_image_landmarks,
            body_model,
            opt,
            ocam,
            vposer,
            model_loss,
            face=face,
            hands=hands,
            lr=lr,
            print_every=print_every,
            writer=writer,
            idx=idx,
            interpenetration=interpenetration,
            )
    print("Start Optimizing Camera")
    opt = optimizer(list(ocam.parameters()) + list(model_loss.parameters()), lr=lr)
    body_model, ocam, idx = run(
            int(num_runs / 4),
            landmarks,
            pose_image_landmarks,
            face_image_landmarks,
            body_model,
            opt,
            ocam,
            vposer,
            model_loss,
            face=face,
            hands=hands,
            lr=lr,
            print_every=print_every,
            writer=writer,
            idx=idx,
            interpenetration=interpenetration,
            )
    opt = optimizer(list(body_model.parameters()) + list(model_loss.parameters()), lr=lr)
    print("Start Optimizing Body Pose")
    body_model, ocam, idx = run(
            int(num_runs / 4),
            landmarks,
            pose_image_landmarks,
            face_image_landmarks,
            body_model,
            opt,
            ocam,
            vposer,
            model_loss,
            face=face,
            hands=hands,
            lr=lr,
            print_every=print_every,
            writer=writer,
            idx=idx,
            interpenetration=interpenetration,
            )

    writer.close()
    body_pose_params = body_model.body_pose
    return body_model, body_pose_params, ocam



def video_opt_loop(
        data,
        body_model,
        vposer,
        num_runs,
        previous_model=None,
        face=False,
        hands=False,
        lr=1e-3,
        cam_type="perspective",
        print_every=200,
        interpenetration=False,
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
    model_loss = rtsmplx.loss.ModelLoss()
    itern = 0
    for param in body_model.parameters():
        if itern in [2, 10]:
            param.requires_grad = True
        else:
            param.requires_grad = False
        itern += 1
    writer = SummaryWriter()

    idx = 0

    opt = optimizer(list(ocam.parameters()) + list(body_model.parameters()) + list(model_loss.parameters()), lr=lr)
    print("Start Optimizing Body Pose")
    body_model, ocam, idx = run(
            num_runs,
            landmarks,
            pose_image_landmarks,
            face_image_landmarks,
            body_model,
            opt,
            ocam,
            vposer,
            model_loss,
            previous_model=previous_model,
            face=face,
            hands=hands,
            lr=lr,
            print_every=print_every,
            writer=writer,
            idx=idx,
            interpenetration=interpenetration,
            )

    writer.close()
    body_pose_params = body_model.body_pose
    return body_model, body_pose_params, ocam


def get_mesh(body_model, body_pose):
    faces = body_model.faces.reshape(-1, 3).detach().cpu().numpy()
    vertices = (
            body_model.forward(body_pose=body_pose).vertices.reshape(-1, 3).detach().cpu().numpy()
            )
    tri_mesh = trimesh.base.Trimesh(vertices=vertices, faces=faces)
    return tri_mesh


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
