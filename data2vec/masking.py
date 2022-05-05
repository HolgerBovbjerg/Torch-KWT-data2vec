"""
Transforms and masking strategies for Self Supervised Pretraining of Vision models.
All codes copied from below repos:
https://github.com/microsoft/unilm/tree/master/beit
https://github.com/rwightman/pytorch-image-models/tree/master/timm
https://github.com/facebookresearch/deit
"""
import warnings
import random
import math
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from timm.data.constants import (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,
                                 IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)
from timm.data.transforms import RandomResizedCropAndInterpolation
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


class MIMTransform(object):
    """
    Masked Image Modeling transforms based on BEiT. copied from https://github.com/microsoft/unilm/tree/master/beit
    """

    def __init__(self, patch_size=16, num_patches=14, num_mask_patches=120, input_size=224, min_num_patches=16,
                 max_num_patches=196,
                 interpolation="bicubic", imagenet_mean_and_std=False):
        imagenet_default_mean_and_std = imagenet_mean_and_std  # cfg.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_mask_patches = num_mask_patches
        self.input_size = input_size
        assert self.patch_size * self.num_patches == self.input_size
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        self.interpolation = interpolation

        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolation(size=self.input_size, interpolation=self.interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        self.masked_position_generator = ImageMaskingGenerator(
            self.num_patches, num_masking_patches=self.num_mask_patches,
            max_num_patches=self.max_num_patches,
            min_num_patches=self.min_num_patches,
        )

    def __call__(self, image):
        return self.common_transform(image), self.masked_position_generator()


class ImageMaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = torch.zeros(self.get_shape(), dtype=torch.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


class AudioMaskingGenerator:
    def __init__(self,
                 mask_prob: float,
                 mask_length: int,
                 attention_mask=None,
                 min_masks: int = 0):
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.attention_mask = attention_mask
        self.min_masks = min_masks

    def __call__(self, shape):
        batch_size, audio_size = shape
        mask = _compute_mask_indices((batch_size, audio_size),
                                     self.mask_prob,
                                     self.mask_length,
                                     self.attention_mask,
                                     self.min_masks)
        mask = torch.from_numpy(mask)
        return mask


def generate_masked_tensor(input_tensor, mask, fill=0):
    masked_tensor = torch.zeros(input_tensor.size(), device=input_tensor.device) + fill
    masked_tensor[mask] = input_tensor[mask]
    return masked_tensor


if __name__ == "__main__":
    audio_mask_generator = AudioMaskingGenerator(mask_prob=0.065,
                                                 mask_length=10,
                                                 attention_mask=None,
                                                 min_masks=1)
    print("Done!")
