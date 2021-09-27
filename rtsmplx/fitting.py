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
        interpenetration=False,
        ):
    pose_mapping = rtsmplx.lm_joint_mapping.get_lm_mapping()
    """
    pose_image_landmarks = torch.cat(
        (pose_image_landmarks, face_image_landmarks), dim=0
    )
    """

    def closure():
        if body == True:
            if vposer==None:
                body_pose = body_model.body_pose
            else:
                body_model.body_pose = nn.Parameter(vposer.decode(body_model.latent_j)["pose_body"])
                body_pose = vposer.decode(body_model.latent_j)["pose_body"]
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

        elbow_knee_prior = elbow_knee_prior_loss(body_pose)

        # Create the search tree
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
            pen_loss = interpenetration_loss(search_tree, pendistance, filter_faces)

        opt.zero_grad()
        loss_pred = loss(
                pose_loss_pred,
                face_loss_pred,
                hands_loss_pred,
                body_pose_param,
                body_pose_prior=body_pose_prior,
                elbow_knee_prior=elbow_knee_prior,
                pen_loss=pen_loss,
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
                interpenetration=interpenetration,
                )
        idx = i

    return body_model, ocam, idx


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
    opt = optimizer(ocam.parameters(), lr=lr)
    body_model, ocam, idx = run(
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
            writer=writer,
            idx=idx,
            interpenetration=interpenetration,
            )
    print(idx)
    opt = optimizer(body_model.parameters(), lr=lr)
    print("Start Optimizing Body Pose")
    body_model, ocam, idx = run(
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
            writer=writer,
            interpenetration=interpenetration,
            idx=idx,
            )
    print("Start Optimizing Camera")
    opt = optimizer(ocam.parameters(), lr=lr)
    body_model, ocam, idx = run(
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
            writer=writer,
            interpenetration=interpenetration,
            idx=idx,
            )
    opt = optimizer(body_model.parameters(), lr=lr)
    print("Start Optimizing Body Pose")
    body_model, ocam, idx = run(
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
            writer=writer,
            interpenetration=interpenetration,
            idx=idx,
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


def interpenetration_loss(search_tree, pen_distance, filter_faces):
    pen_loss = 0.0
    batch_size = projected_joints.shape[0]
    triangles = torch.index_select(
        body_model_output.vertices, 1,
        body_model_faces).view(batch_size, -1, 3, 3)

    with torch.no_grad():
        collision_idxs = search_tree(triangles)

    if collision_idxs.ge(0).sum().item() > 0:
        pen_loss = torch.sum(pen_distance(triangles, collision_idxs))

    return pen_loss


def pose_loss(joint_coords_2d, landmarks_2d):
    loss_func = l1loss()
    return loss_func(joint_coords_2d, landmarks_2d)


def face_loss(bary_coords_2d, landmarks_2d):
    loss_func = l1loss()
    return loss_func(bary_coords_2d, landmarks_2d)

def elbow_knee_prior_loss(body_pose):
    # 4 5 elbow
    # 18 19 knee
    ek_id = [4,5,18,19]
    ek_prior = torch.sum(torch.exp(body_pose[:, ek_id]))
    return None

def loss(
        pose_loss,
        face_loss,
        hands_loss,
        body_pose,
        body_pose_prior=None,
        elbow_knee_prior=None,
        pen_loss = None,
        regu=[1e-4, 1e-4],
        ):
    loss_val = pose_loss + face_loss + hands_loss

    # priors
    if len(regu) == 1:
        regu = [regu, regu, regu]
    if body_pose_prior:
        loss_val = loss_val + regu[0] * body_pose_prior
    if elbow_knee_prior:
        loss_val = loss_val + regu[1] * elbow_knee_prior
    if pen_loss:
        loss_val = loss_val + regu[2] * pen_loss
    return loss_val


def mse_loss():
    return nn.MSELoss()


def l1loss():
    return nn.L1Loss()


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
