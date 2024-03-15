import os
import wandb
import json
from PIL import Image

'''
    *************************************** ATTENTION ***************************************
    1. To run different guide modes, classifier_scale, and positive_label, you can modify the following variables.
    2. If the script is accidentally terminated, you just re-run the script and **it will skip the already done evaluations**.
    3. Before using this script, be sure to have the following files:
        - model_path, classifier_path
        - ref images (we will use {refdir}/{dataset}_test_{label}.npz for evaluation, run image2npy.py first to generate these files)
    4. The results will be saved in the workdir, and the sampled images will be uploaded to wandb.
    *****************************************************************************************
'''

guide_modes = [
    'dynamic-two-0.5*a*(1-a)', 'dynamic-two-0.1*a*(1-a)'
]
manifold_strength = [10, 50, 100, 200]
dynamic_strength = [100, 200, 500, 1000]

loss = ['cosine', 'l2']
target = ['./datasets/celeba_hq_256/03437.jpg', './datasets/celeba_hq_256/14255.jpg', './datasets/celeba_hq_256/21177.jpg']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default=1, help='dataset name')
parser.add_argument('--cuda_id', type=int, default=1, help='dataset name')
parser.add_argument('--eta', type=float, default=1, help='dataset name')
args = parser.parse_args()

classifier_path = './ckpts/celebA/model_ir_se50.pth'
model_id = 'google/ddpm-celebahq-256'
target = args.target
num_samples = 1000
batch_size = 32
timesteps = 50
diffusion_steps = 1000
noise_schedule = 'linear'
learn_sigma = False
use_ddim = True
eta=args.eta

CUDA_VISIBLE_DEVICES = args.cuda_id

workdir = '/home/linhw/code/guided-diffusion/haowei/3.15+celeba'

# wandb configs
# project_name = f"sweep={dataset}+date=3.13+eta={eta}"
entity_name = "guided-diffusion"
project_name = 'celebA-sweep-3.15'

'''
    The following part does not need to be modified
'''

os.makedirs(workdir, exist_ok=True)


def collect_results(logdir, guide_mode, scale, target):

    result, image = None, None

    if os.path.exists(f'{logdir}/log.json') and os.path.exists(img):

        with open(f"{logdir}/log.json") as f:
            data = json.load(f)
            fid = data['FID']
            sfid = data['sFID']
            kid = data['kid']
            faceid = data['faceid']

        result = [guide_mode, scale, target, kid, fid, sfid, faceid]
        image = Image.open(img)

    else:
        print("WARNING: no log.json or image found")
        
    return result, image

def log_wandb(lst):
    keys = ['image', 'mode', 'scale', 'label', 'kid', 'fid', 'sfid', 'faceid']
    return {k: v for k, v in zip(keys, lst)}


run = None

for guide_mode in guide_modes:

    for scale in zip(manifold_strength, dynamic_strength):

        if 'manifold' in guide_mode:
            scale = scale[0]
        else:
            scale = scale[1]
        
        for losstype in loss:
            
            if run is not None:
                wandb.finish()
                run = None
            
            targetname = target.split('/')[-1].split('.')[0]
            
            run = wandb.init(
                project=project_name,
                name=f"guide_mode={guide_mode}+scale={scale}+target={targetname}+losstype={losstype}+eta={eta}",
                entity=entity_name
            )

            print(f'guide_mode={guide_mode}, scale={scale}, target={targetname}, losstype={losstype}, eta={eta}')

            # some necessary paths
            logdir = f'{workdir}/target={targetname}+losstype={losstype}+eta={eta}/mode={guide_mode}+scale={float(scale)}'

            npz = f'{logdir}/mode={guide_mode}+scale={float(scale)}/samples_{num_samples}x256x256x3.npz'
            img = f'{logdir}/mode={guide_mode}+scale={float(scale)}/samples_{num_samples}x256x256x3.png'
            log = f'{logdir}/log.json'
            ref_batch = f'/home/linhw/code/guided-diffusion/evaluations/ref/celeba_hq_256.npz'

            # check if the evaluation is already done before
            if os.path.exists(log) and os.path.exists(img):

                result, image = collect_results(logdir, guide_mode, scale, targetname)

                if result is not None and image is not None:
                    result = [wandb.Image(image, caption=f"target={targetname}+losstype={losstype}+eta={eta}/mode={guide_mode}+scale={float(scale)}")] + result
                    run.log(log_wandb(result))

                continue
            
            # check if there are already sampling results
            if not os.path.exists(npz) or not os.path.exists(img):
                
                os.system(
                    f'''
                        CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} proxychains python scripts/classifier_sample.py \
                            --progress True \
                            --log_dir "{logdir}" \
                            --timestep_respacing 50 \
                            --use_ddim True \
                            --batch_size {batch_size} \
                            --num_samples {num_samples} \
                            --positive_label {target} \
                            --classifier_scale {scale} \
                            --guide_mode "{guide_mode}" \
                            --eta {eta} \
                            --model_id {model_id} \
                            --classifier_path {classifier_path} \
                            --image_size 256 \
                            --num_channels 128 \
                            --num_res_blocks 3 \
                            --diffusion_steps 1000 \
                            --noise_schedule linear \
                            --iteration 1
                ''')

            # check if the evaluation is already done before
            if not (os.path.exists(log) and os.path.exists(img)):
                # run the evaluation
                os.system(
                    f'''
                    CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python ./evaluations/face_eval.py \
                    --ref_batch "{ref_batch}" --sample_batch "{npz}" --target {target} --output_path "{logdir}/log.json"
                    ''')

            # check if the evaluation is already done 
            if os.path.exists(log) and os.path.exists(img):

                result, image = collect_results(logdir, guide_mode, scale, targetname)

                if result is not None and image is not None:
                    result = [wandb.Image(image, caption=f"target={targetname}+losstype={losstype}+eta={eta}/mode={guide_mode}+scale={float(scale)}")] + result
                    run.log(log_wandb(result))

                continue