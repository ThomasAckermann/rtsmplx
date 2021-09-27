import torch
import torch.nn as nn
import rtsmplx.utils as utils


class ModelLoss(nn.Module):
    def __init__(self):# , search_tree, pen_distance, filter_faces):
        super(ModelLoss, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        pose_prior_weight = 1e-3 * torch.ones(1)
        self.register_buffer("pose_prior_weight", pose_prior_weight)
        elbow_knee_weight = 3e-3 * torch.ones(1)
        self.register_buffer("elbow_knee_weight", elbow_knee_weight)
        self.register_buffer("joint_dist_weight", torch.ones(1))
        self.robustifier = utils.robustifier_func(rho=100)

    """
    def interpenetration_loss(self, search_tree, pen_distance, filter_faces):
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
    """

    def pose_loss(self, joints_2d, landmarks_2d):
        loss_func = nn.L1Loss()
        return loss_func(joints_2d, landmarks_2d)

    """
    def face_loss(self):
        loss_func = nn.L1Loss()
        return loss_func(self.bary_coords_2d, self.landmarks_2d)
    """

    def elbow_knee_prior_loss(self, body_pose):
        # 4 5 elbow
        # 18 19 knee
        ek_id = [4,5,18,19]
        ek_prior = torch.sum(torch.exp(body_pose[:, ek_id]))
        return ek_prior

    def body_pose_prior(self, body_pose):
        return torch.linalg.norm(body_pose).to(device=self.device)

    def forward(self, body_pose, joints_2d, landmarks_2d):
        l1loss = nn.L1Loss()
        pose_loss = torch.sum(self.robustifier(joints_2d - landmarks_2d)) * self.joint_dist_weight
        face_loss = 0.0
        hands_loss = 0.0
        body_pose_prior = self.body_pose_prior(body_pose)
        elbow_knee_prior = self.elbow_knee_prior_loss(body_pose)

        # pen_loss = self.interpenetration_loss(self.search_tree, self.pen_distance, self.filter_faces)

        loss_val = pose_loss + face_loss + hands_loss
        if body_pose_prior:
            loss_val = loss_val + self.pose_prior_weight * body_pose_prior
        if elbow_knee_prior:
            loss_val = loss_val + self.elbow_knee_weight * elbow_knee_prior
        # if pen_loss:
        #     loss_val = loss_val + self.regularization[2] * pen_loss
        return loss_val

