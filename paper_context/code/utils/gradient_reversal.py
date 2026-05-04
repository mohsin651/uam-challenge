"""Gradient reversal layer (GRL) for adversarial training.

Reference: Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation"
(ICML 2015). https://arxiv.org/abs/1409.7495

Forward pass: identity (x → x).
Backward pass: gradient gets multiplied by -lambda (sign flipped, magnitude scaled).

When this is placed between a feature extractor and a domain/camera classifier,
the classifier learns to predict camera ID from features (positive grad path),
but the feature extractor is pushed to make features the classifier CAN'T
distinguish (negative grad through GRL). Result: camera-invariant features.

Usage:
    cam_logits = camera_classifier(grad_reverse(features, lambda_=0.1))
    cam_loss = F.cross_entropy(cam_logits, camera_labels)
    total_loss = id_loss + cam_loss   # cam_loss's gradient flows back AS NEGATIVE through GRL
"""
import torch
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)  # identity (preserves grad graph)

    @staticmethod
    def backward(ctx, grad_output):
        # Flip sign and scale; second return is None (lambda_ is a scalar, no grad needed)
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with the given scale."""
    return _GradReverse.apply(x, lambda_)
