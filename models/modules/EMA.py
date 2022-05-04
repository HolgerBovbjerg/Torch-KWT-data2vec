import torch
import torch.nn as nn


class EMA:
    """
    Modified version of class fairseq.models.ema.EMA.
    """

    def __init__(self, model: nn.Module, device=None, skip_keys=None, ema_decay=0.999):
        self.model = model
        self.model.requires_grad_(False)
        self.model.to(device)
        self.device = device
        self.skip_keys = skip_keys or set()
        self.decay = ema_decay
        self.num_updates = 0

    def step(self, new_model: nn.Module):
        ema_state_dict = {}
        ema_params = self.model.state_dict()
        for key, param in new_model.state_dict().items():
            ema_param = ema_params[key].float()
            if key in self.skip_keys:
                ema_param = param.to(dtype=ema_param.dtype).clone()
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(param.to(dtype=ema_param.dtype), alpha=1 - self.decay)
            ema_state_dict[key] = ema_param
        self.model.load_state_dict(ema_state_dict, strict=False)
        self.num_updates += 1

    def restore(self, model: nn.Module):
        d = self.model.state_dict()
        model.load_state_dict(d, strict=False)
        return model

    def _set_decay(self, decay):
        self.decay = decay

    def get_decay(self):
        return self.decay

    @staticmethod
    def get_annealed_rate(start, end, curr_step, total_steps):
        r = end - start
        pct_remaining = 1 - curr_step / total_steps
        return end - r * pct_remaining
