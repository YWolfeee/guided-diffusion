from argparse import Namespace
from typing import Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from guided_diffusion import dist_util
from .unet import EncoderUNetModel

def get_cond_fn(classifier: EncoderUNetModel, args: Namespace
                ) -> Tuple[Union[nn.Module, None], dict]:
    """Obtain the conditional classifier guidance function based on args."""
    model_kwargs = {"guide_mode": args.guide_mode,
                    "classifier_scale": args.classifier_scale,
                    "positive_label": args.positive_label}
    model_kwargs["y"] = args.positive_label * torch.ones((args.batch_size,), device=dist_util.dev(), dtype=torch.int)
    if args.guide_mode in ["None", "none", None] or args.classifier_scale == 0:
        return None, model_kwargs
    def cond_fn(x, t, y=None, **kwargs):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale
    if args.guide_mode in ["classifier", "guide_x0", "manifold", 'unbiased']:
        return cond_fn, model_kwargs
    else:
        raise ValueError(f"Unknown guide mode: {args.guide_mode}")
    