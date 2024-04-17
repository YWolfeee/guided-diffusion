"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import time

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import imageio
np.random.seed(0)
th.manual_seed(0)

from guided_diffusion import dist_util, logger
from guided_diffusion.conditional_fun import get_cond_fn, get_target_cond_fn
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
    ResNet18_32x32,
)

from PIL import Image
import math
from matplotlib import pyplot as plt

def main():
    args = create_argparser().parse_args()

    if args.guide_mode in ["None", "none", None]:
        args.guide_mode = None
        args.classifier_scale = 0.0
        args.positive_label = None
        logger.log("No classifier guidance will be used.")
    assert args.class_cond is False, "We focus on the setting where the diffusion mode is unconditional and the guidance is accomplished via an additional classifier."
    pipe = "ddim" if args.use_ddim else "ddjm" if args.use_ddjm else "ddpm"
    args.log_dir = os.path.join(args.log_dir, f"steps={args.timestep_respacing}+pipe={pipe}+iter={args.iteration}+mode={args.guide_mode}+scale={args.classifier_scale}+shrink={args.shrink_cond_x0}")
    logger.configure(dir=args.log_dir)

    dist_util.setup_dist()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if not args.model_id:   
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    args.has_time = True
    if args.classifier_path == "" or args.positive_label == "" or args.guide_mode is None:
        print("No guidance considered. Skipping classifier loading.")
        classifier = None
        cond_fn = None
        model_kwargs = {"guide_mode": args.guide_mode,
                    "classifier_scale": args.classifier_scale,
                    "positive_label": args.positive_label}
    elif args.positive_label.endswith(".jpg") or args.positive_label.endswith(".png"):
        pass
        # guide with images
        from guided_diffusion.celebA.faceid import load_arcface, arcface_forward_path
        classifier = load_arcface(args.classifier_path, dist_util.dev())
        tar_feat = arcface_forward_path(classifier, args.positive_label, dist_util.dev())
        model_kwargs = {"guide_mode": args.guide_mode,
                    "classifier_scale": args.classifier_scale,
                    "positive_label": args.positive_label}
        cond_fn = get_target_cond_fn(classifier, tar_feat, args)
    elif args.classifier_path.endswith("pt") or args.classifier_path.endswith("pth"):
        if args.classifier_path.split("/")[-1].startswith("not"):
            classifier = ResNet18_32x32()
            classifier.load_state_dict(
                dist_util.load_state_dict(args.classifier_path, map_location="cpu"))
            args.has_time = False
        else:
            classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
            # from PyTorch_CIFAR10.cifar10_models import resnet
            # classifier = resnet.resnet50()
            classifier.load_state_dict(
                dist_util.load_state_dict(args.classifier_path, map_location="cpu"))
        classifier.to(dist_util.dev())
        if args.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()
        cond_fn, model_kwargs = get_cond_fn(classifier, args)
    else:
        raise ValueError(f"Unknown classifier path: {args.classifier_path}")

    def model_fn(x, t, y=None,**kwargs):
        # assert y is not None
        return model(x, t, y if args.class_cond else None)
        # return model(x, t, y if args.guide_mode is not None else None)

    assert 1-(args.use_ddim & args.use_ddjm)
    sample_fn = (diffusion.ddim_sample_loop if args.use_ddim else (
                 diffusion.ddjm_sample_loop if args.use_ddjm else 
                 diffusion.p_sample_loop))
    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        start = time.time()
        
        sample, traj = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            progress=args.progress,
            eta=args.eta,
            iteration=args.iteration,
            shrink_cond_x0=args.shrink_cond_x0,
            score_norm=args.score_norm
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        '''
            save the trajectory of samples and the predicted x0
        '''

        if args.plot_traj:
            traj = {
                'samples': [((x['sample'] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(0,2,3,1).contiguous() for x in traj],
                'pred_x0': [((x['pred_xstart'] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(0,2,3,1).contiguous()  for x in traj]
            }
            samples_pil = [[Image.fromarray(y.cpu().numpy()) for y in time] for time in traj['samples']]
            x0_pil = [[Image.fromarray(y.cpu().numpy()) for y in time] for time in traj['pred_x0']]
            
            for i in range(len(samples_pil)):
                if i % 50 == 0 or i == len(samples_pil) - 1:
                    save_images(samples_pil[i], os.path.join(logger.get_dir(), f"sample_{i}.png"))
                    save_images(x0_pil[i], os.path.join(logger.get_dir(), f"x0_{i}.png"))
            
            # gif_samples = [Image.open(os.path.join(logger.get_dir(), f"sample_{i}.png")) for i in range(len(samples_pil))]
            # gif_x0 = [Image.open(os.path.join(logger.get_dir(), f"x0_{i}.png")) for i in range(len(x0_pil))]
            # imageio.mimsave(os.path.join(logger.get_dir(), f"sample_traj.gif"), gif_samples, duration=5)
            # imageio.mimsave(os.path.join(logger.get_dir(), f"x0_traj.gif"), gif_x0, duration=5)

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if 'y' in model_kwargs:
            classes = model_kwargs["y"]
            gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        end = time.time()
        logger.log(f"created {len(all_images) * args.batch_size} samples, {end - start:.01f} sec per batch")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if 'y' in model_kwargs:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    else:
        label_arr = None
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        np.savez(out_path, arr, label_arr)
        img_path = out_path.replace('.npz', '.png')
        save_images(arr, img_path)
        logger.log(f"saving to {img_path}")

    dist.barrier()
    logger.log("sampling complete")
    # clear all torch gpu memory usage
    th.cuda.empty_cache()


    if args.ref_batch is not None:
        eval_args = argparse.Namespace()
        eval_args.ref_batch = args.ref_batch
        eval_args.sample_batch = out_path
        eval_args.classifier_path = args.test_classifier_path if args.test_classifier_path else args.classifier_path
        eval_args.label = int(args.positive_label) if args.guide_mode is not None else None
        eval_args.output_path = os.path.join(logger.get_dir(), 'log.json')
        from evaluations.evaluator import main as evaluator_main
        logger.log(f"Running evaluator with args: {eval_args}")
        evaluator_main(eval_args)
    else:
        logger.log("No reference batch provided, skipping evaluation")


def save_images(images: np.ndarray, 
                path_name: str, 
                labels=None,
                dpi: int=300,
                ):
    
    # images = [Image.fromarray(image) for image in images]
    images = images[:min(64, len(images))]
    length = math.ceil(math.sqrt(len(images)))
    fig, axs = plt.subplots(length, length, figsize=(16, 16))
    for i in range(length):
        for j in range(length):
            if i*length+j >= len(images):
                continue
            axs[i, j].imshow(images[i*length+j])
            if labels is not None:
                axs[i, j].set_title(int(labels[i*length+j].item()))
            axs[i, j].set_axis_off()
    fig.savefig(path_name, dpi=dpi, bbox_inches='tight')

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        use_ddjm=False,
        model_path="",
        log_dir="tmp",
        classifier_path="",
        guide_mode="None",
        classifier_scale=0.0,
        positive_label="None",
        progress=False,
        eta=0.0,
        ref_batch=None,
        test_classifier_path="",
        model_id=None,
        iteration=10,
        shrink_cond_x0=True,    # whether to shrink the score of x0 by at
        faceid_loss_type='cosine',
        face_image1_id='00000',
        face_image2_id='00000',
        face_image3_id='00000',
        plot_args=False,
        score_norm=1e09,
        plot_traj=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
