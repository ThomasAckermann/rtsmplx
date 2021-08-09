import smplx


class BodyModel:
    """smpl-x body model

    Keyword arguments:
    model_path -- string of path to model
    """

    def __init__(self, model_path, num_expression_coeffs=10, gender="neutral"):
        super(BodyModel, self).__init__()
        self.model_path = model_path
        self.model = smplx.body_models.create(self.model_path, "smplx")
