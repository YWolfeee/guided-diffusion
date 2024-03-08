import os

guide_modes = [
'dynamic-full-0.5*a*(1-a)', 'dynamic-full-0.1*a*(1-a)', 'dynamic-two-0.5*a*(1-a)', 'dynamic-two-0.1*a*(1-a)', 'dynamic-two-0.5*a-0.5*(1-a)', 'dynamic-two-0.1*a-0.1*(1-a)', 'dynamic-one-0.5*(1-a)', 'dynamic-one-0.1*(1-a)'
]

classifier_scale = [0.1, 0.5, 1.0, 2.0] # for classifier
else_scale = [5, 10, 20, 50] # for else
positive_label = [1]

# 15 x 4 x 3 = 180

model_path="/home/linhw/code/guided-diffusion/ckpts/mnist/ema_0.9999_100000.pt"
classifier_path="/home/linhw/code/guided-diffusion/ckpts/mnist_classifier/model099999.pt"

for guide_mode in guide_modes[-2:]:

    for scale in zip(classifier_scale, else_scale):

        if guide_mode == 'classifier':
            scale = scale[0]
        else:
            scale = scale[1]
        
        for label in positive_label:

            print(f'guide_mode={guide_mode}, scale={scale}, label={label}')

            os.system(
                f'''
                CUDA_VISIBLE_DEVICES=6 python scripts/classifier_sample.py \
                --timestep_respacing 50 \
                --use_ddim True \
                --log_dir "/home/linhw/code/guided-diffusion/haowei/3.8-mnist/guide_mode={guide_mode}-scale={scale}-label={label}" \
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
