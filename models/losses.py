import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, weights=None):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(weight=weights)

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        dice = 1 - (2 * (probs * one_hot).sum() + 1e-6) / ((probs + one_hot).sum() + 1e-6)
        return self.alpha * ce + (1 - self.alpha) * dice.mean()
