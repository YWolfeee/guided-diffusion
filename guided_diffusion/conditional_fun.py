from argparse import Namespace
from typing import Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from guided_diffusion import logger

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
        # def xt_score_fn(xt, t, y=None, **kwargs):
        #     # print("In side")
        #     model_func = kwargs['out_func']
        #     with torch.enable_grad():
        #         xt_in = xt.detach().requires_grad_(True)
        #         x0 = model_func(x=xt_in, t=t)['pred_xstart']
        #         logits = classifier(x0, torch.zeros_like(t))
        #         log_probs = F.log_softmax(logits, dim=-1)
        #         selected = log_probs[range(len(logits)), y.view(-1).long()]
        #         return torch.autograd.grad(selected.sum(), xt_in)[0] * args.classifier_scale
        # return the classifier directly and compute gradient in the loop
        def model_fun(x, t, y=None, **kwargs):
            logits = classifier(x, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1).long()]
            return selected * args.classifier_scale
        return model_fun, model_kwargs
    def cond_fn(x, t, y=None, **kwargs):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            if args.has_time:
                logits = classifier(x_in, t)
            else:
                logits = classifier(x_in)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1).long()]

            if 'zero_order' in kwargs.keys() and kwargs['zero_order']:
                return F.softmax(logits, dim=-1)[range(len(logits)), y.view(-1).long()] * args.classifier_scale
            else:
                return torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale, selected
    return cond_fn, model_kwargs
    # if args.guide_mode in ["classifier", "guide_x0", "manifold", 'unbiased', "resample"]:
    # else:
    # raise ValueError(f"Unknown guide mode: {args.guide_mode}")
def get_target_cond_fn(classifier, tar_feat, args: Namespace):
    from guided_diffusion.celebA.faceid import arcface_forward
    from PIL import Image
    import numpy as np

    with torch.no_grad():
        image1 = Image.open(f'/home/linhw/code/guided-diffusion/datasets/celeba_hq_256/{args.face_image1_id}.jpg').convert('RGB').resize([256, 256], resample=2)
        data1 = torch.tensor(np.array(image1).transpose(2, 0, 1), device='cuda').unsqueeze(0) / 127.5 - 1

        image2 = Image.open(f'/home/linhw/code/guided-diffusion/datasets/celeba_hq_256/{args.face_image2_id}.jpg').convert('RGB').resize([256, 256], resample=2)
        data2 = torch.tensor(np.array(image2).transpose(2, 0, 1), device='cuda').unsqueeze(0) / 127.5 - 1

        image3 = Image.open(f'/home/linhw/code/guided-diffusion/datasets/celeba_hq_256/{args.face_image3_id}.jpg').convert('RGB').resize([256, 256], resample=2)
        data3 = torch.tensor(np.array(image3).transpose(2, 0, 1), device='cuda').unsqueeze(0) / 127.5 - 1
            
        output1 = arcface_forward(classifier, data1)
        output2 = arcface_forward(classifier, data2)
        output3 = arcface_forward(classifier, data3)
        target_feat1 = output1
        target_feat2 = output2
        target_feat3 = output3

    def cond_fn (x, t, y=None, **kwargs):

        with torch.enable_grad():
            
            if args.guide_mode != 'freedom':
                x = x.detach().requires_grad_(True)
            
            feat = arcface_forward(classifier, x)
            
            if args.faceid_loss_type == 'cosine':
                dist = torch.cosine_similarity(feat, tar_feat, dim=-1)
            elif args.faceid_loss_type == 'l2':

                dist1 = - torch.linalg.norm(feat - target_feat1, dim=-1) \
                    + (- torch.linalg.norm(feat - target_feat2, dim=-1)) \
                    + (- torch.linalg.norm(feat - target_feat3, dim=-1))
                # dist1 = - torch.linalg.norm(feat - target_feat1, dim=-1)

                # dist2 = - torch.linalg.norm(data1 - x_in, dim=-1).mean(dim=-1).mean(dim=-1)
                dist2 = (- torch.linalg.norm(data1 - x, dim=-1) + \
                        (- torch.linalg.norm(data2 - x, dim=-1)) + \
                        (- torch.linalg.norm(data3 - x, dim=-1)) ).mean(dim=-1).mean(dim=-1)

                logger.info('{} {}'.format(dist1.mean().item(), dist2.mean().item()))

                dist = dist1 + dist2
            else:
                raise NotImplementedError
            logger.info("{}".format(dist.mean().item()))
            
            if args.guide_mode == 'freedom':
                return dist.sum() * args.classifier_scale
            else:
                return torch.autograd.grad(dist.sum(), x)[0] * args.classifier_scale
    return cond_fn