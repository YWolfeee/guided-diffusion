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
'classifier', 'manifold', 'dynamic-fonly', 'dynamic-full-0.5*a*(1-a)', 'dynamic-full-0.1*a*(1-a)', 'dynamic-two-0.5*a*(1-a)', 'dynamic-two-0.1*a*(1-a)', 'dynamic-two-0.5*a-0.5*(1-a)', 'dynamic-two-0.1*a-0.1*(1-a)', 'dynamic-one-0.5*(1-a)', 'dynamic-one-0.1*(1-a)', 'dynamic-nog-0.5*a', 'dynamic-nog-0.1*a', 'dynamic-nog-0.5', 'dynamic-nog-0.1'
]
# guide_modes = ['dynamic-nog-0.1*a']

classifier_scale = [0.5, 1.0, 2.0] # for classifier
manifold_scale = [5, 20, 50]
dynamic_scale = [20, 50, 100]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--label', type=int, default=1, help='dataset name')
parser.add_argument('--cuda_id', type=int, default=1, help='dataset name')
args = parser.parse_args()

positive_label = [args.label]
num_samples = 2048
batch_size = 256
timesteps = 50

CUDA_VISIBLE_DEVICES = args.cuda_id
dataset = "mnist"   # dataset name

refdir = "/home/linhw/code/guided-diffusion/evaluations/ref"    # path to reference images (will use {refdir}/{dataset}_test_{label}.npz for evaluation)
if dataset == "mnist":
    model_path="/home/linhw/code/guided-diffusion/ckpts/mnist/ema_0.9999_100000.pt"         # path to the diffusion model
    classifier_path="/home/linhw/code/guided-diffusion/ckpts/mnist_classifier/model099999.pt"   # path to the classifier
    eval_classifier="/home/linhw/code/guided-diffusion/evaluations/eval_cls/mnist_test/model_34500_0.9909.pth"  # evaluation classifier
    workdir = "/home/linhw/code/guided-diffusion/haowei/3.11+mnist+eta=1"  # path to save any output files
    eta=1
elif dataset == 'cifar':
    model_path="/home/linhw/code/guided-diffusion/ckpts/cifar10/ema_0.9999_200000.pt"         # path to the diffusion model
    classifier_path="/home/linhw/code/guided-diffusion/ckpts/cifar_classifier/model099999.pt"   # path to the classifier
    workdir = "/home/linhw/code/guided-diffusion/haowei/3.8-cifar"  # path to save any output files
    eval_classifier="/home/linhw/code/guided-diffusion/evaluations/eval_cls/cifar_test/model_89500_0.8720999956130981.pth"

# wandb configs
project_name = f"sweep={dataset}+date=3.10+eta=1"
entity_name = "guided-diffusion"

'''
    The following part does not need to be modified
'''

os.makedirs(workdir, exist_ok=True)


def collect_results(logdir, guide_mode, scale, label):

    result, image = None, None

    if os.path.exists(f'{logdir}/log.json') and os.path.exists(img):

        with open(f"{logdir}/log.json") as f:
            data = json.load(f)
            validity = data['validity']
            fid = data['fid']
            sfid = data['sfid']
            precision = data['precision']
            recall = data['recall']
            is_ = data['inception_score']

        result = [guide_mode, scale, label, validity, fid, sfid, precision, recall, is_]
        image = Image.open(img)

    else:
        print("WARNING: no log.json or image found")
        
    return result, image

def log_wandb(lst):
    keys = ['image', 'mode', 'scale', 'label', 'validity', 'fid', 'sfid', 'precision', 'recall', 'is']
    return {k: v for k, v in zip(keys, lst)}


run = None

for guide_mode in guide_modes:

    for scale in zip(classifier_scale, manifold_scale, dynamic_scale):

        if guide_mode == 'classifier':
            scale = scale[0]
        elif 'manifold' in guide_mode:
            scale = scale[1]
        else:
            scale = scale[2]
        
        for label in positive_label:
            
            if run is not None:
                wandb.finish()
                run = None
            
            run = wandb.init(
                project=project_name,
                name=f"guide_mode={guide_mode}+scale={scale}+label={label}",
                entity=entity_name
            )

            print(f'guide_mode={guide_mode}, scale={scale}, label={label}')

            # some necessary paths
            logdir = f'{workdir}/label={label}/mode={guide_mode}+scale={scale}'
            if not os.path.exists(logdir):
                logdir = f'{workdir}/label={label}/mode={guide_mode}+scale={scale}.0'

            npz = f'{logdir}/samples_{num_samples}x32x32x3.npz'
            img = f'{logdir}/samples_{num_samples}x32x32x3.png'
            ref_batch = f'{refdir}/{dataset}_test_{label}.npz'

            # check if the evaluation is already done before
            if os.path.exists(f'{logdir}/log.json') and os.path.exists(img):

                result, image = collect_results(logdir, guide_mode, scale, label)

                if result is not None and image is not None:
                    result = [wandb.Image(image, caption=f"guide_mode={guide_mode}, scale={scale}, label={label}")] + result
                    run.log(log_wandb(result))

                continue
            
            # check if there are already sampling results
            if not os.path.exists(npz) or not os.path.exists(img):
                
                os.system(
                    f'''
                    CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python scripts/classifier_sample.py \
                    --timestep_respacing {timesteps} \
                    --use_ddim True \
                    --log_dir "{workdir}/label={label}" \
                    --batch_size {batch_size} \
                    --num_samples {num_samples} \
                    --positive_label {label} \
                    --classifier_scale {scale} \
                    --guide_mode "{guide_mode}" \
                    --model_path {model_path} \
                    --classifier_path {classifier_path} \
                    --image_size 32 \
                    --num_channels 128 \
                    --num_res_blocks 3 \
                    --eta {eta} \
                    --diffusion_steps 1000 \
                    --noise_schedule linear \
                    
                ''')

            # run the evaluation
            os.system(
                f'''
                CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python ./evaluations/evaluator.py \
                {ref_batch} "{npz}" --classifier_path {eval_classifier} --label {label} --output_path "{logdir}/log.json"
                ''')

            # check if the evaluation is already done before
            if os.path.exists(f'{logdir}/log.json') and os.path.exists(img):

                result, image = collect_results(logdir, guide_mode, scale, label)

                if result is not None and image is not None:
                    result = [wandb.Image(image, caption=f"guide_mode={guide_mode}, scale={scale}, label={label}")] + result
                    run.log(log_wandb(result))

                continue
