"""Focal loss for edge labels."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossSaturnCloud(nn.Module):

    def __init__(self, alpha=None, gamma=4):
        super(FocalLossSaturnCloud, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        if isinstance(self.alpha, torch.Tensor) and self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss
