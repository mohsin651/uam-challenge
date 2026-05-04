"""Exponential Moving Average of model parameters.

Standard technique: maintain a 'shadow' copy of model weights that updates as
    shadow = decay * shadow + (1 - decay) * current_weights
after every optimizer step. At inference time, swap shadow in place of the
raw weights. Usually buys +0.5-1.5% mAP on ReID-style tasks essentially for free.

Decay 0.9999 ≈ effective window of ~10000 steps (all of training for our setup).
Decay 0.999 ≈ ~1000 steps.

Use:
    ema = ModelEMA(model, decay=cfg.SOLVER.EMA_DECAY)
    ...
    # in training loop, after scaler.step(optimizer):
    ema.update(model)
    ...
    # at save-time:
    ema.apply_shadow(model)
    torch.save(model.state_dict(), path)
    ema.restore(model)
"""
import torch


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().float()
        # also track buffers (BN running stats etc.) — important for stability
        self.shadow_buffers = {}
        for name, buf in model.named_buffers():
            self.shadow_buffers[name] = buf.data.clone().float()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(d).add_(param.data.float(), alpha=1.0 - d)
        for name, buf in model.named_buffers():
            if name in self.shadow_buffers:
                # for BN running stats use hard copy (they're already EMA-like)
                self.shadow_buffers[name].copy_(buf.data.float())

    def apply_shadow(self, model):
        """Swap model weights with EMA shadow; keep originals in backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.dtype))

    def restore(self, model):
        """Restore original (non-EMA) weights from backup."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
