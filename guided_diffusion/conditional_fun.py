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
    model_kwargs["y"] = int(args.positive_label) * torch.ones((args.batch_size,), device=dist_util.dev(), dtype=torch.int)
    if args.guide_mode in ["None", "none", None]:
        return None, model_kwargs
    elif args.guide_mode == 'freedom':
        def xt_score_fn(xt, t, y=None, **kwargs):
            # print("In side")
            model_func = kwargs['out_func']
            with torch.enable_grad():
                xt_in = xt.detach().requires_grad_(True)
                x0 = model_func(x=xt_in, t=t)['pred_xstart']
                logits = classifier(x0, torch.zeros_like(t))
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1).long()]
                return torch.autograd.grad(selected.sum(), xt_in)[0] * args.classifier_scale
        return xt_score_fn, model_kwargs
    def cond_fn(x, t, y=None, **kwargs):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1).long()]
            return torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale
    return cond_fn, model_kwargs
    # if args.guide_mode in ["classifier", "guide_x0", "manifold", 'unbiased', "resample"]:
    # else:
    # raise ValueError(f"Unknown guide mode: {args.guide_mode}")

def get_target_cond_fn(classifier, tar_feat, args: Namespace):
    from guided_diffusion.celebA.faceid import arcface_forward
    def cond_fn (x, t, y=None, **kwargs):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            feat = arcface_forward(classifier, x_in)
            if args.faceid_loss_type == 'cosine':
                dist = torch.cosine_similarity(feat, tar_feat, dim=-1)
            elif args.faceid_loss_type == 'l2':
                dist = - torch.linalg.norm(feat - tar_feat, dim=-1)
            else:
                raise NotImplementedError
            print(dist.mean().item())
            return torch.autograd.grad(dist.sum(), x_in)[0] * args.classifier_scale
    return cond_fn
