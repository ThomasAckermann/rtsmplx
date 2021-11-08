import torc0*h
import torch.nn as nn


class ModelLoss(nn.Module):
    def __init__(self):# , search_tree, pen_distance, filter_faces):
        super(ModelLoss, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        pose_weight = 1 * torch.ones(1)
        self.register_buffer("pose_weight", pose_weight.to(device=self.device))
        pose_prior_weight = 0* 5e-3 * torch.ones(1)
        self.register_buffer("pose_prior_weight", pose_prior_weight.to(device=self.device))
        elbow_knee_weight = 0*5e-3 * torch.ones(1)
        self.register_buffer("elbow_knee_weight", elbow_knee_weight.to(device=self.device))
        previous_image_weight = 0*1e-2 * torch.ones(1)
        self.register_buffer("previous_image_weight", previous_image_weight.to(device=self.device))
        previous_cam_weight = 0*1e-1 * torch.ones(1)
        self.register_buffer("previous_cam_weight", previous_cam_weight.to(device=self.device))
        silhouette_weight = 0*1 * torch.ones(1) # 1e-2
        self.register_buffer("silhouette_weight", silhouette_weight.to(device=self.device))


    def previous_image_prior(self, body_pose, previous_model):
        previous_pose = previous_model.body_pose.detach()
        l2loss = nn.MSELoss()
        return l2loss(body_pose, previous_pose)

    def previous_cam_prior(self, new_camera, old_camera):
        l2loss = nn.MSELoss()
        cam_loss = (
                l2loss(new_camera.rotation, old_camera.rotation.detach())
                + l2loss(new_camera.translation, old_camera.translation.detach())
                + l2loss(new_camera.rotation, old_camera.scale.detach())
                )

        return cam_loss

    def silhouette_loss(self, silhouette_image, silhouette_prediction):
        silhouette_image = silhouette_image / 255
        silhouette_prediction = silhouette_prediction / 255
        img_shape = silhouette_image.shape
        silhouette_prediction = silhouette_prediction.reshape(img_shape)[:,:,0].to(dtype=torch.float)
        silhouette_image = silhouette_image[:,:,0].to(dtype=torch.float)

        lossf = nn.MSELoss(reduction="mean")
        sil_loss = lossf(silhouette_image, silhouette_prediction)
        return sil_loss


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
        ek_prior = torch.sum(torch.exp(1*body_pose[:, ek_id]))
        return ek_prior

    def body_pose_prior(self, body_pose):
        return torch.linalg.norm(body_pose).to(device=self.device)

    def forward(self, body_pose, joints_2d, landmarks_2d, camera, previous_model=None, previous_cam=None, silhouette_image=None, silhouette_prediction=None):
        pose_loss = self.pose_loss(joints_2d, landmarks_2d)
        face_loss = 0.0
        hands_loss = 0.0
        body_pose_prior = self.body_pose_prior(body_pose)
        elbow_knee_prior = self.elbow_knee_prior_loss(body_pose)

        loss_val = self.pose_weight * pose_loss + face_loss + hands_loss
        if body_pose_prior:
            loss_val = loss_val + self.pose_prior_weight * body_pose_prior
        if elbow_knee_prior:
            loss_val = loss_val + self.elbow_knee_weight * elbow_knee_prior
        if previous_model:
            previous_image_prior = self.previous_image_prior(body_pose, previous_model)
            loss_val = loss_val + self.previous_image_weight * previous_image_prior
        if previous_cam:
            previous_cam_prior = self.previous_cam_prior(camera, previous_cam)
            loss_val = loss_val + self.previous_cam_weight * previous_cam_prior
        if (silhouette_image != None) and (silhouette_prediction != None):
            silhouette_loss = self.silhouette_loss(silhouette_image, silhouette_prediction)
            loss_val = loss_val + self.silhouette_weight * silhouette_loss

        # if pen_loss:
        # pen_loss = self.interpenetration_loss(self.search_tree, self.pen_distance, self.filter_faces)
        #     loss_val = loss_val + self.regularization[2] * pen_loss
        return loss_val

