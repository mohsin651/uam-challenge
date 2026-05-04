"""ArcFace logits computation — angular-margin classifier for ReID.

Paper: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
Recognition" (CVPR 2019). https://arxiv.org/abs/1801.07698

Key idea: instead of plain linear logits W @ x, compute cos(theta + margin)
on the TRUE class only. Encourages tighter intra-class clustering and bigger
inter-class angular separation. Standard in person ReID.

This file provides a HELPER FUNCTION (not a Module). The classifier weights
remain owned by the model — we just use them as the W matrix here, normalized.
This keeps the optimizer/state-dict clean (no extra params introduced).

Usage in model forward (training):
    if cfg.MODEL.ID_LOSS_TYPE == 'arcface' and label is not None:
        cls_score = arcface_logits(
            feat, self.classifier.weight, label,
            scale=cfg.MODEL.ARCFACE_S, margin=cfg.MODEL.ARCFACE_M,
        )
    else:
        cls_score = self.classifier(feat)
"""
import math
import torch
import torch.nn.functional as F


def arcface_logits(feat, weight, label, scale: float = 64.0, margin: float = 0.50):
    """ArcFace-modified logits.

    feat:    (B, D) features (will be L2-normalized)
    weight:  (num_classes, D) classifier weights (will be L2-normalized)
    label:   (B,) integer class labels in [0, num_classes-1]
    scale:   logit scaling factor (s in paper, typically 30-64)
    margin:  additive angular margin in radians (m in paper, typically 0.3-0.5)

    Returns logits of shape (B, num_classes) — feed directly into CE loss.
    """
    # L2 normalize feature and weight matrix → cos_theta is in [-1, 1]
    feat_norm = F.normalize(feat, p=2, dim=1)
    weight_norm = F.normalize(weight, p=2, dim=1)
    cos_theta = feat_norm @ weight_norm.t()                  # (B, C)

    # Numerical safety for sqrt
    cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    sin_theta = torch.sqrt(1.0 - cos_theta.pow(2))

    cos_m = math.cos(margin)
    sin_m = math.sin(margin)
    # cos(theta + m) = cos_theta * cos_m - sin_theta * sin_m
    cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

    # When theta + m > pi, cos(theta+m) wraps around; use easier substitution
    # following the original ArcFace implementation
    th = math.cos(math.pi - margin)
    mm = math.sin(math.pi - margin) * margin
    cos_theta_m = torch.where(cos_theta > th, cos_theta_m, cos_theta - mm)

    # Apply margin only on the TRUE class entries
    one_hot = torch.zeros_like(cos_theta)
    one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
    logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
    return logits * scale
