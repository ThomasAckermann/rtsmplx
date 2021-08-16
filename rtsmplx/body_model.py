import smplx
import torch


class BodyModel(smplx.body_models.SMPLX):
    """smpl-x body model

    Keyword arguments:
    model_path -- string of path to model
    """

    def __init__(self, model_path, num_expression_coeffs=10, gender="neutral"):
        super(BodyModel, self).__init__(model_path=model_path)
        self.bary_coords = self.lmk_bary_coords
        self.bary_faces_idx = self.lmk_faces_idx.type(torch.long)
        self.faces = self.faces_tensor.type(torch.long)
        self.bary_faces = self.faces[self.bary_faces_idx]
        self.vertices = self.v_template
        self.bary_vertices = self.vertices[self.bary_faces]

    def get_joints(self):
        forward_out = self.forward()
        joints = forward_out.joints
        return joints
