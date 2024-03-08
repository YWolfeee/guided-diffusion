import numpy as np
import os

guide_modes = [
'classifier', 'manifold', 'guide_x0', 'dynamic-full-0.5*a*(1-a)', 'dynamic-full-0.1*a*(1-a)', 'dynamic-two-0.5*a*(1-a)', 'dynamic-two-0.1*a*(1-a)', 'dynamic-two-0.5*a-0.5*(1-a)', 'dynamic-two-0.1*a-0.1*(1-a)', 'dynamic-one-0.5*(1-a)', 'dynamic-one-0.1*(1-a)', 'dynamic-nog-0.5*a', 'dynamic-nog-0.1*a', 'dynamic-nog-0.5', 'dynamic-nog-0.1'
]

classifier_scale = [0.1, 0.5, 1.0, 2.0] # for classifier
else_scale = [5, 10, 20, 50] # for else
positive_label = [1, 4, 8]

classifier_path = '/home/linhw/code/guided-diffusion/evaluations/eval_cls/mnist_test/model_34500_0.9909.pth'

for guide_mode in guide_modes[1:]:

    for scale in zip(classifier_scale, else_scale):

        if guide_mode == 'classifier':
            scale = scale[0]
        else:
            scale = scale[1]
        
        for label in positive_label:

            print(f'guide_mode={guide_mode}, scale={scale}, label={label}')
            
            logdir = f'/home/linhw/code/guided-diffusion/haowei/3.8-mnist/guide_mode={guide_mode}-scale={scale}-label={label}/mode={guide_mode}+scale={scale}.0'

            npz = f'{logdir}/samples_10240x32x32x3.npz'

            if os.path.exists(f'{logdir}/log.json') or not os.path.exists(npz):
                continue

            ref_batch = f'/home/linhw/code/guided-diffusion/evaluations/ref/mnist_test_{label}.npz'
            
            os.system(
                f'''
                CUDA_VISIBLE_DEVICES=0 python ./evaluator.py \
                {ref_batch} {npz} --classifier_path {classifier_path} --label {label} --output_path {logdir}/log.json
                ''')
