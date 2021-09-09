import smplx
import torch
import torch.nn as nn


class BodyModel(smplx.body_models.SMPLX):
    """smpl-x body model

    Keyword arguments:
    model_path -- string of path to model
    """

    def __init__(self, model_path, gender="neutral"):
        super(BodyModel, self).__init__(model_path=model_path)

        self.gender = gender
        self.bary_coords = self.lmk_bary_coords
        self.bary_faces_idx = self.lmk_faces_idx.type(torch.long)
        self.faces = self.faces_tensor.type(torch.long)
        self.bary_faces = self.faces[self.bary_faces_idx]
        self.vertices = self.v_template
        self.bary_vertices = self.vertices[self.bary_faces]
        self.body_pose = self.body_pose

    def get_joints(self, body_pose=None):
        if body_pose == None:
            body_pose = self.body_pose
        forward_out = self.forward(body_pose=body_pose)
        joints = forward_out.joints.reshape(-1, 3)
        return joints
