import torch
import torch.nn as nn


class Loss(nn.Module):
    """Loss function"""

    def __init__(self):
        super(Loss, self).__init__()

