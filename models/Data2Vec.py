import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.EMA import EMA
from data2vec.masking import generate_masked_tensor


class Data2Vec(nn.Module):
    """
    Data2Vec main module.
    Args:
         encoder (nn.Module)
         cfg (omegaconf.DictConfig)
    """
    MODALITIES = ['vision', 'text', 'audio']

    def __init__(self,
                 encoder,
                 modality,
                 model_embed_dim,
                 ema_decay, ema_end_decay,
                 ema_anneal_end_step,
                 average_top_k_layers,
                 normalize_targets,
                 **kwargs):
        super(Data2Vec, self).__init__()
        self.encoder = encoder
        assert modality in self.MODALITIES
        self.modality = modality
        self.embed_dim = model_embed_dim
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step
        self.average_top_k_layers = average_top_k_layers
        self.normalize_targets = normalize_targets
        self.__dict__.update(kwargs)

        self.ema = EMA(self.encoder)  # EMA acts as the teacher
        self.regression_head = self._build_regression_head()

    def _build_regression_head(self):
        if self.modality == 'text':
            embed_dim = self.embed_dim
            curr_dim = embed_dim
            projections = []
            for i in range(self.cfg.model.head_layers - 1):
                next_dim = embed_dim * 2 if i == 0 else curr_dim
                projections.append(nn.Linear(curr_dim, next_dim))
                projections.append(nn.GELU())
                curr_dim = next_dim

            projections.append(nn.Linear(curr_dim, embed_dim))
            return nn.Sequential(*projections)

        if self.modality in ['audio', 'vision']:
            return nn.Linear(self.embed_dim, self.embed_dim)

    def ema_step(self):
        """
        One EMA step for the offline model until the ending decay value is reached
        """
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema.step(self.encoder)


    def forward(self, src, trg=None, mask=None, **kwargs):
        """
        Data2Vec forward method.
        Args:
            src: src tokens (masked inputs for training, and inference)
            trg: trg tokens (unmasked inputs for training but left as `None` otherwise)
            mask: bool masked indices, Note: if a modality requires the inputs to be masked before forward this param
            has no effect. (see the Encoder for each modality to see if it uses mask or not)
        Returns:
            Either encoder outputs or a tuple of encoder + EMA outputs
        """
        encoder_out, student_hidden_states = self.encoder(src, mask=mask, output_hidden_states=True)
        if trg is None:
            return encoder_out
        x = student_hidden_states[-1]
        with torch.no_grad():
            self.ema.model.eval()

            _, teacher_hidden_states = self.ema.model(trg, mask=None, output_hidden_states=True)

            y = teacher_hidden_states[-self.average_top_k_layers:]
            if self.modality in ['vision', 'text']:  # Follow the same layer normalization procedure for text and vision
                y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
                y = sum(y) / len(y)
                if self.normalize_targets:
                    y = F.layer_norm(y.float(), y.shape[-1:])

            elif self.modality == 'audio':  # Use instance normalization for audio
                y = [F.instance_norm(tl.float()) for tl in y]
                y = sum(y) / len(y)
                if self.normalize_targets:
                    y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        x = x[mask]
        y = y[mask]

        x = self.regression_head(x)

        return x, y
