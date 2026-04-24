"""Circle Loss for deep metric learning (ReID).

Reference: Sun et al., "Circle Loss: A Unified Perspective of Pair Similarity
Optimization" (CVPR 2020). https://arxiv.org/abs/2002.10857

Key idea: instead of the fixed margin of Triplet loss, Circle Loss rescales
each pair's gradient by how far that pair is from its decision boundary,
giving stronger gradients to pairs that need more work. Strongly
outperforms Triplet on most ReID benchmarks.

Drop-in replacement for `triplet(feat, target)[0]`: call like
    criterion = CircleLoss(margin=0.25, gamma=256)
    loss = criterion(feat, label)

Defaults from the paper: m=0.25, gamma=256.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    def __init__(self, margin: float = 0.25, gamma: float = 256.0):
        super().__init__()
        self.m = margin
        self.gamma = gamma
        # optimal similarities (same as in the paper)
        self.O_p = 1 + self.m       # positive optimum (we want sp ≥ O_p)
        self.O_n = -self.m          # negative optimum (we want sn ≤ O_n)
        self.delta_p = 1 - self.m   # positive margin (boundary)
        self.delta_n = self.m       # negative margin (boundary)

    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # feat: (B, D). Normalize to unit hypersphere first so similarities are
        # in [-1, 1].
        feat = F.normalize(feat, p=2, dim=1)
        B = feat.size(0)

        # Full pairwise cosine similarity
        sim = feat @ feat.t()  # (B, B)

        # Masks: positive pairs (same label, not self), negative pairs (diff label).
        label = label.view(-1, 1)
        is_same = label.eq(label.t())
        eye = torch.eye(B, dtype=torch.bool, device=feat.device)
        pos_mask = is_same & ~eye
        neg_mask = ~is_same

        # Collect positive and negative similarities.
        # Any batch without at least one positive+negative pair is degenerate;
        # in that case return zero. (Won't happen with RandomIdentitySampler.)
        if not pos_mask.any() or not neg_mask.any():
            return feat.sum() * 0.0

        # Per-sample Circle Loss: compute for each anchor separately so the
        # logsumexp is grouped correctly.
        losses = []
        for i in range(B):
            sp = sim[i][pos_mask[i]]
            sn = sim[i][neg_mask[i]]
            if sp.numel() == 0 or sn.numel() == 0:
                continue
            # Self-adaptive weights (detached — they just scale the gradient)
            alpha_p = F.relu(self.O_p - sp.detach())
            alpha_n = F.relu(sn.detach() - self.O_n)
            # Exponents
            logit_p = -self.gamma * alpha_p * (sp - self.delta_p)
            logit_n =  self.gamma * alpha_n * (sn - self.delta_n)
            # softplus(lse_n + lse_p) is the scalar loss for this anchor
            loss_i = F.softplus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
            losses.append(loss_i)

        return torch.stack(losses).mean() if losses else feat.sum() * 0.0
