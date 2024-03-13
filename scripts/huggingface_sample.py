import numpy as np
import json
import torch
import os
from guided_diffusion.celebA.ddpm import FaceDDIMPipeline
from diffusers import DDIMScheduler, UNet2DModel


def main():
    
    model_id = "google/ddpm-celebahq-256"
    # load model and scheduler
    model = UNet2DModel.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
    scheduler = DDIMScheduler.from_pretrained(model_id)

    config = {
        'sample_mode': 'guide_x0',
        'guide_strength': 4,
        'seed': 0,
        'arcface_path': '/home/linhw/code/guided-diffusion/guided_diffusion/celebA/resnet18_110.pth',
        'target_path': '/home/linhw/code/guided-diffusion/guided_diffusion/celebA/celeba_hq_256/04539.jpg'
    }
    from argparse import Namespace
    config = Namespace(**config)
    
    pipeline = FaceDDIMPipeline(unet=model, scheduler=scheduler,).to('cuda')

    pipeline.prepare_arcface(config, target=config.target_path)
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    pipeline_output = pipeline(
        batch_size=1, 
        generator=torch.manual_seed(config.seed),
    )

    images = [x for x in pipeline_output.images]
    
    for x in images:
        x.save('guide_x0.png')

if __name__ == '__main__':
    main()