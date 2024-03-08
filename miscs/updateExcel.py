import openpyxl
import os
import numpy as np
import json

guide_modes = [
'classifier', 'manifold', 'guide_x0', 'dynamic-full-0.5*a*(1-a)', 'dynamic-full-0.1*a*(1-a)', 'dynamic-two-0.5*a*(1-a)', 'dynamic-two-0.1*a*(1-a)', 'dynamic-two-0.5*a-0.5*(1-a)', 'dynamic-two-0.1*a-0.1*(1-a)', 'dynamic-one-0.5*(1-a)', 'dynamic-one-0.1*(1-a)', 'dynamic-nog-0.5*a', 'dynamic-nog-0.1*a', 'dynamic-nog-0.5', 'dynamic-nog-0.1'
]

classifier_scale = [0.1, 0.5, 1.0, 2.0] # for classifier
else_scale = [5, 10, 20, 50] # for else
positive_label = [1, 4, 8]

title = ['mode', 'scale', 'label', 'validity', 'fid', 'sfid', 'precision', 'recall', 'is']

wb = openpyxl.Workbook()
sheet = wb.active

sheet.append(title)

for mode in guide_modes:

    for scale in zip(classifier_scale, else_scale):
        
        for label in positive_label:

            if mode == 'classifier':
                scale = scale[0] if type(scale) == tuple else scale
                logdir = f'/home/linhw/code/guided-diffusion/haowei/3.8-mnist/guide_mode={mode}-scale={scale}-label={label}/mode={mode}+scale={scale}/log.json'
            else:
                scale = scale[1] if type(scale) == tuple else scale
                logdir = f'/home/linhw/code/guided-diffusion/haowei/3.8-mnist/guide_mode={mode}-scale={scale}-label={label}/mode={mode}+scale={scale}.0/log.json'
            

            try:
                with open(logdir, 'r') as f:
                    data = json.load(f)
                    validity = data['validity']
                    fid = data['fid']
                    sfid = data['sfid']
                    precision = data['precision']
                    recall = data['recall']
                    is_ = data['inception_score']
            except:
                continue

            print(f'guide_mode={mode}, scale={scale}, label={label}')

            sheet.append([mode, scale, label, validity, fid, sfid, precision, recall, is_])

wb.save('/home/linhw/code/guided-diffusion/haowei/3.8-mnist/eval.xlsx')
