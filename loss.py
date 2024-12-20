import torch.nn as nn

class weighted_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, weight):
        return ((input - target)**2 * weight).sum() / weight.sum()
