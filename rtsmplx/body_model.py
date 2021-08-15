import smplx
import torch


class BodyModel:
    """smpl-x body model

    Keyword arguments:
    model_path -- string of path to model
    """

    def __init__(self, model_path, num_expression_coeffs=10, gender="neutral"):
        super(BodyModel, self).__init__()
        self.model_path = model_path
        self.model = smplx.body_models.create(self.model_path, "smplx")
        self.bary_coords = self.model.lmk_bary_coords
        self.bary_faces_idx = self.model.lmk_faces_idx.type(torch.long)
        self.faces = self.model.faces_tensor.type(torch.long)
        self.bary_faces = self.faces[self.bary_faces_idx]
        self.vertices = self.model.v_template
        self.bary_vertices = self.vertices[self.bary_faces]
