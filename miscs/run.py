import os

guide_modes = [
'dynamic-full-0.5*a*(1-a)', 'dynamic-full-0.1*a*(1-a)', 'dynamic-two-0.5*a*(1-a)', 'dynamic-two-0.1*a*(1-a)', 'dynamic-two-0.5*a-0.5*(1-a)', 'dynamic-two-0.1*a-0.1*(1-a)', 'dynamic-one-0.5*(1-a)', 'dynamic-one-0.1*(1-a)'
]

classifier_scale = [0.1, 0.5, 1.0, 2.0] # for classifier
else_scale = [5, 10, 20, 50] # for else
positive_label = [1, 4, 8]

# 8 * 3 * 3 = 96 / 12 card = 8 iter

home_path = '.'
model_path=f"{home_path}/ckpts/mnist-diffusion-10w/ema_0.9999_100000.pt"
classifier_path=f"{home_path}/ckpts/mnist-classifier-10w/model099999.pt"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--mode_id', type=int, default=0)
parser.add_argument('--label_id', type=int, default=0)
args = parser.parse_args()

# print args in key, value style
print("args:")
for key, value in vars(args).items():
    print(f"{key}: {value}")

gpu_id = args.gpu_id
guide_mode = guide_modes[args.mode_id]
label = positive_label[args.label_id]
for scale in zip(classifier_scale, else_scale):

    if guide_mode == 'classifier':
        scale = scale[0]
    else:
        scale = scale[1]
    

    print(f'guide_mode={guide_mode}, scale={scale}, label={label}')

    os.system(
        f'''
        CUDA_VISIBLE_DEVICES={gpu_id} python scripts/classifier_sample.py \
        --timestep_respacing 50 \
        --use_ddim True \
        --log_dir "sweep/mnist+label={label}/guide_mode={guide_mode}-scale={scale}-label={label}" \
        --batch_size 256 \
        --num_samples 10240 \
        --positive_label {label} \
        --classifier_scale {scale} \
        --guide_mode "{guide_mode}" \
        --model_path {model_path} \
        --classifier_path {classifier_path} \
        --image_size 32 \
        --num_channels 128 \
        --num_res_blocks 3 \
        --diffusion_steps 1000 \
        --noise_schedule linear \
        
    ''')
