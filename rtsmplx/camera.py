import torch

class Camera:
    def __init__(self):
        self.camera_matrix = torch.zeros(3, 4)

    def calibrate(self, depth_map, landmarks):
        pass

    


