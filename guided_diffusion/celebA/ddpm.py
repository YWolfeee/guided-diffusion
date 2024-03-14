from dataclasses import dataclass
from diffusers import DDPMPipeline, DDIMPipeline
import torch
import torch.nn as nn
import PIL
from PIL import Image
import numpy as np
from typing import List, Optional, Tuple, Union

from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput
from torchmetrics.image.inception import InceptionScore
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler

from guided_diffusion.celebA.faceid import load_arcface, arcface_forward, cosine

class CustomPipelineOutput(BaseOutput):
    images: Union[torch.Tensor, PIL.Image.Image]  # images generated by the model

class FaceDDIMPipeline(DDIMPipeline):

    def __init__(self, unet: UNet2DModel, scheduler: DDIMScheduler):
        super().__init__(unet, scheduler)

    @torch.no_grad()
    def prepare_arcface(self, config, target='/home/linhw/code/guided-diffusion/guided_diffusion/celebA/celeba_hq_256/00000.jpg'):
        self.sample_mode = config.sample_mode
        self.guide_strength = config.guide_strength
        self.arcface_model = load_arcface(config.arcface_path,self.device)

        self.arcface_model.to(self.device)
        image = Image.open(target).convert('RGB')
        data = torch.tensor(np.array(image).transpose(2, 0, 1), device=self.device).unsqueeze(0) / 127.5 - 1
        output = arcface_forward(self.arcface_model, data)
        self.target_feat = output
    
    @torch.enable_grad()
    def get_guide_score(self, x):
        x_in = x.detach().requires_grad_(True)
        feat = arcface_forward(self.arcface_model, x_in)
        prob = torch.log(cosine(feat, self.target_feat))
        return torch.autograd.grad(prob.sum(), x_in)[0] * self.guide_strength
    
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[CustomPipelineOutput, Tuple]:
        
        unet = self.unet
        scheduler = self.scheduler
        
        # Sample gaussian noise to begin loop
        if isinstance(unet.config.sample_size, int):
            image_shape = (
                batch_size,
                unet.config.in_channels,
                unet.config.sample_size,
                unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, unet.config.in_channels, *unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            x = randn_tensor(image_shape, generator=generator)
            x = x.to(self.device)
        else:
            x = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        scheduler.set_timesteps(num_inference_steps)

        # if 'first_order' in self.sample_mode:
        #     accepetance = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
            
        for t in self.progress_bar(scheduler.timesteps):
            # 1. predict noise model_output
            model_output = unet(x, t).sample

            # 2. compute previous image: x_t -> x_t-1
            scheduler_output = scheduler.step(model_output, t, x, generator=generator)
            new_x = scheduler_output.prev_sample
            pred_x0 = scheduler_output.pred_original_sample

            # 3. Add guidance according to guidance type
            if self.sample_mode == 'default':
                x = new_x   # standard unconditional sampling

            elif self.sample_mode == 'guide_x0':
                score = self.get_guide_score(pred_x0)
                sqrt_acum = scheduler.alphas_cumprod[t-1]**0.5 if t else 1
                x = new_x + self.guide_strength * sqrt_acum * score
              
        if output_type == "pil":
            x = self.image_tensor_to_pil(x)

        return CustomPipelineOutput(images=x)

    def image_tensor_to_pil(self, images):
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return self.numpy_to_pil(images)
    
  

if __name__ == '__main__':
    model_id = "https://huggingface.co/google/ddpm-celebahq-256"
    # load model and scheduler
    ddpm = DDIMPipeline.from_pretrained(model_id).to('cuda')  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

    # run pipeline in inference (sample random noise and denoise)
    for step in [50, 100, 200, 500, 1000]:
        generator = torch.Generator().manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=1000).images[0]

        # save image
        image.save("images/generated_image_{}.png".format(step))