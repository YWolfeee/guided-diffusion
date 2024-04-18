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

guide_modes = ['manifold', 'freedom']
cls_scales = [10, 20, 30]
iterations = [1]
# steps = [200, 50]
recurrent = [1, 5, 10]
shrinks = [False]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--label', type=str, default='1')
parser.add_argument('--setting', type=str, default='recurrent', help='dataset name')
parser.add_argument('--cuda_id', type=int, default=1, help='dataset name')
parser.add_argument('--dataset', type=str, default='cifar', help='dataset name')
parser.add_argument('--shrink_cond_x0', action='store_true', help='whether to shrink the score of x0')
parser.add_argument('--timesteps', type=int, default=1000, help='number of timesteps')
args = parser.parse_args()

positive_label = [int(x) for x in args.label.split("+")]
num_samples = 2048
timesteps = args.timesteps
eta=1
setting = args.setting
shrink_cond_x0 = args.shrink_cond_x0
use_ddim = True
use_ddjm = not use_ddim
batch_size = 256

CUDA_VISIBLE_DEVICES = args.cuda_id
dataset = args.dataset   # dataset name

refdir = "/home/linhw/code/guided-diffusion/evaluations/ref"    # path to reference images (will use {refdir}/{dataset}_test_{label}.npz for evaluation)
if dataset == "mnist":
    model_path="/home/linhw/code/guided-diffusion/ckpts/mnist/ema_0.9999_100000.pt"         # path to the diffusion model
    classifier_path="/home/linhw/code/guided-diffusion/ckpts/mnist_classifier/model099999.pt"   # path to the classifier
    eval_classifier="/home/linhw/code/guided-diffusion/evaluations/eval_cls/mnist_test/model_34500_0.9909.pth"  # evaluation classifier
    workdir = f"/home/linhw/code/guided-diffusion/haowei/3.17+mnist+eta={eta}+timestep={timesteps}+setting={setting}"  # path to save any output files
elif dataset == 'cifar':
    model_path="/home/linhw/code/guided-diffusion/ckpts/cifar10/ema_0.9999_200000.pt"         # path to the diffusion model
    # classifier_path="/home/linhw/code/guided-diffusion/ckpts/cifar_classifier/model099999.pt"   # path to the classifier
    classifier_path="/home/linhw/code/not_best_epoch96_acc0.9470.ckpt"
    workdir = f"/home/linhw/code/guided-diffusion/haowei/4.18+cifar+eta={eta}+timestep={timesteps}+setting={setting}"  # path to save any output files
    eval_classifier="nateraw/vit-base-patch16-224-cifar10"

# wandb configs
project_name = f"sweep={dataset}+date=4.19+recurrent+unshrink"
entity_name = "llm-selection"

'''
    The following part does not need to be modified
'''

os.makedirs(workdir, exist_ok=True)

def collect_results(logdir, guide_mode, scale, label, setting, iteration, shrink_x0, timesteps, recurrent_):

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

        result = [guide_mode, scale, label, validity, fid, sfid, precision, recall, is_, setting, iteration, shrink_x0, timesteps, recurrent_]
        image = Image.open(img)

    else:
        print("WARNING: no log.json or image found")
        
    return result, image

def log_wandb(lst):
    keys = ['image', 'mode', 'scale', 'label', 'validity', 'fid', 'sfid', 'precision', 'recall', 'is', 'setting', 'iteration', 'shrink_x0', 'timesteps', 'recurrent']
    return {k: v for k, v in zip(keys, lst)}


run = None


for label_ in positive_label:

    for guide_mode in guide_modes:

        guide_mode_type = 'classifier' if 'classifier' in guide_mode else 'manifold' if 'manifold' in guide_mode else 'dynamic' if 'dynamic' in guide_mode else 'freedom' if 'freedom' in guide_mode else 'None'

        if guide_mode_type == 'None':
            label = -1
        else:
            label = label_

        for scale in cls_scales:

            for recurrent_ in recurrent:

                for iteration in iterations:
                
                    if run is not None:
                        wandb.finish()
                        run = None
                    
                    run = wandb.init(
                        project=project_name,
                        name=f"guide_mode={guide_mode}+scale={scale}+label={label}+setting={setting}+iter={iteration}+shrink_cond_x0={shrink_cond_x0}+timesteps={timesteps}+recurrent={recurrent_}",
                        entity=entity_name
                    )

                    print(f"guide_mode={guide_mode}+scale={scale}+label={label}+setting={setting}+iter={iteration}+shrink_cond_x0={shrink_cond_x0}+timesteps={timesteps}+recurrent={recurrent_}")

                    pipe = 'ddjm' if use_ddjm else 'ddim'
                    # some necessary paths
                    logdir = f'{workdir}/label={label}/setting={setting}/shrink_cond_x0={shrink_cond_x0}/recurrent={recurrent_}/steps={timesteps}+pipe={pipe}+iter={iteration}+mode={guide_mode}+scale={float(scale)}+shrink={shrink_cond_x0}'

                    npz = f'{logdir}/samples_{num_samples}x32x32x3.npz'
                    img = f'{logdir}/samples_{num_samples}x32x32x3.png'


                    if label != -1:
                        if dataset == 'cifar':
                            correct = [7, 1, 0, 2, 3, 4, 5, 6, 8, 9]
                            ref_label = correct.index(label)
                            print(ref_label)
                        else:
                            ref_label = label
                        ref_batch = f'{refdir}/{dataset}_test_{ref_label}.npz'
                    else:
                        ref_batch = f'{refdir}/{dataset}_test.npz'

                    # check if the evaluation is already done before
                    if os.path.exists(f'{logdir}/log.json') and os.path.exists(img):
                        
                        result, image = collect_results(logdir, guide_mode, scale, label, setting, iteration, shrink_cond_x0, timesteps, recurrent_)

                        if result is not None and image is not None:
                            result = [wandb.Image(image, caption=f"guide_mode={guide_mode}+scale={scale}+label={label}+setting={setting}+iter={iteration}+shrink_cond_x0={shrink_cond_x0}+timesteps={timesteps}+recurrent={recurrent_}")] + result
                            run.log(log_wandb(result))

                        continue

                    # check if there are already sampling results
                    if not os.path.exists(npz) or not os.path.exists(img):
                        
                        os.system(
                            f'''
                            CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python scripts/classifier_sample.py \
                            --timestep_respacing {timesteps} \
                            --use_ddim {use_ddim} \
                            --use_ddjm {use_ddjm} \
                            --log_dir "{workdir}/label={label}/setting={setting}/shrink_cond_x0={shrink_cond_x0}/recurrent={recurrent_}" \
                            --batch_size {batch_size} \
                            --num_samples {num_samples} \
                            --positive_label {label} \
                            --classifier_scale {scale} \
                            --guide_mode "{guide_mode}" \
                            --model_path "{model_path}" \
                            --classifier_path "{classifier_path}" \
                            --image_size 32 \
                            --num_channels 128 \
                            --iteration {iteration} \
                            --num_res_blocks 3 \
                            --eta {eta} \
                            --diffusion_steps 1000 \
                            --noise_schedule linear \
                            --shrink_cond_x0 {shrink_cond_x0} \
                            --recurrent {recurrent_}
                        ''')

                    # check if the evaluation is already done before
                    if not (os.path.exists(f'{logdir}/log.json') and os.path.exists(img)):
                        # run the evaluation
                        os.system(
                            f'''
                            CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python ./evaluations/evaluator.py \
                            "{ref_batch}" "{npz}" --classifier_path {eval_classifier} --label {label} --output_path "{logdir}/log.json"
                            ''')

                    # check if the evaluation is already done before
                    if os.path.exists(f'{logdir}/log.json') and os.path.exists(img):

                        result, image = collect_results(logdir, guide_mode, scale, label, setting, iteration, shrink_cond_x0,timesteps,recurrent_)

                        if result is not None and image is not None:
                            result = [wandb.Image(image, caption=f"guide_mode={guide_mode}+scale={scale}+label={label}+setting={setting}+iter={iteration}+shrink_cond_x0={shrink_cond_x0}+timesteps={timesteps}+recurrent={recurrent_}")] + result
                            run.log(log_wandb(result))

                        continue
