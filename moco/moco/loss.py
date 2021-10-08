import torch
import torch.nn as nn


class WeightedSoftmaxMoCoLoss(nn.Module):
    """
    Apply weights to the sum of exponential in the softmax for
    MoCo contrastive loss.
    """
    def __init__(self):
        super(WeightedSoftmaxMocoLoss, self).__init__()

    def forward(self, logits, weights, dim=1):
        assert logits.shape == weights.shape
        weighted_logsumexp = self.log_sum_exp(logits, weights, dim)
        # exp_neg = torch.sum(torch.exp(logits[:, 1:]) * weights, dim)
        # exp_pos = torch.exp(logits[:, 0])
        # loss = -torch.log(exp_pos) - torch.log(exp_pos + exp_neg)
        loss = -logits[:, 0] + weighted_logsumexp
        loss = loss.mean(0)
        return loss

    def log_sum_exp(self, logits, weights, dim=1):
        m, _ = torch.max(logits, dim=dim, keepdim=True)
        lse = m.squeeze(1) + torch.log(torch.sum(torch.exp(logits-m) * weights,
                                       dim=dim))
        return lse


