import torch
from torch import nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, image):
        return out


class RNN(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # layers
        runit = torch.GRU(input_size=(23,3), hidden_size=20, num_layers=2)

    def forward(self, x):
        out = runit(x)
        return out

