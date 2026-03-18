import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- 1. Focal Loss 实现 --------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        gamma: 聚焦参数，减少易分类样本的权重
        alpha: 类别权重（tensor），若为None则使用均匀权重
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [batch_size, num_classes]
        targets: [batch_size]
        """
        log_probs = F.log_softmax(logits, dim=-1)          # log(softmax)
        probs = torch.exp(log_probs)                       # softmax概率
        # 提取每个样本正确类别的log_prob和prob
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze()
        pt = probs.gather(1, targets.view(-1, 1)).squeeze()

        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss