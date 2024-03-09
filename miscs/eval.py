import numpy as np
import os

guide_modes = [
'classifier', 'manifold', 'guide_x0', 'dynamic-full-0.5*a*(1-a)', 'dynamic-full-0.1*a*(1-a)', 'dynamic-two-0.5*a*(1-a)', 'dynamic-two-0.1*a*(1-a)', 'dynamic-two-0.5*a-0.5*(1-a)', 'dynamic-two-0.1*a-0.1*(1-a)', 'dynamic-one-0.5*(1-a)', 'dynamic-one-0.1*(1-a)', 'dynamic-nog-0.5*a', 'dynamic-nog-0.1*a', 'dynamic-nog-0.5', 'dynamic-nog-0.1'
]

classifier_scale = [0.1, 0.5, 1.0, 2.0] # for classifier
else_scale = [5, 10, 20, 50] # for else
positive_label = [1, 4, 8]

classifier_path = 'ckpts/mnist-test-classifier/model_34500_0.9909.pth'


def extract_specific_kwargs(dirname):
    """Extract specific kwargs from the directory name, handling complex values."""
    # Initial dictionary to hold the extracted values
    kwargs = {}

    # List of keys in the order they appear in the directory name
    keys = ['guide_mode', 'scale', 'label']
    # Create a pattern to match the keys and capture their values
    pattern = r"(?P<guide_mode>guide_mode=[^=]+?(?=-scale=))-(?P<scale>scale=[^=]+?(?=-label=))-(?P<label>label=.+)"
    
    import re
    match = re.search(pattern, dirname)
    if match:
        for key in keys:
            if match.group(key):
                # Remove the key and '=' from the captured value
                value = match.group(key).split('=', 1)[1]
                kwargs[key] = value

    return kwargs

sweep_path = './sweep'
"""Go through all sub-dirs in sweep_path and print specific kwargs for each leaf dir."""
for path, dirs, files in os.walk(sweep_path):
    # If 'dirs' is empty, it means 'path' is a leaf directory
    if dirs:
        continue
    name = "/".join(path.split("/")[:-1])
    kwargs = extract_specific_kwargs(name)
    print(f"Leaf directory: {path}")
    print(f"Specific Kwargs: {kwargs}")
    
    label = kwargs['label']


    npz = os.path.join(path,'samples_10240x32x32x3.npz')
    json = os.path.join(path, 'log.json')

    if os.path.exists(json) or not os.path.exists(npz):
        continue

    ref_batch = f'./evaluations/ref/mnist_train_{label}.npz'
    
    print(
        f'''
        CUDA_VISIBLE_DEVICES=0 python ./evaluator.py \
        {ref_batch} {npz} --classifier_path {classifier_path} --label {label} --output_path {json}
        ''')

import pandas as pd
columns = [
    "guide_mode", "label", "scale", 
    "validity", "fid", "sfid", "precision", "recall", "inception_score",
]

csv = open("./stats.csv", 'w')
csv.write(", ".join(columns) + '\n')
for path, dirs, files in os.walk(sweep_path):
    # If 'dirs' is empty, it means 'path' is a leaf directory
    if dirs:
        continue
    name = "/".join(path.split("/")[:-1])
    kwargs = extract_specific_kwargs(name)
    # print(f"Leaf directory: {path}")
    # print(f"Specific Kwargs: {kwargs}")
    
    label = kwargs['label']

    json = os.path.join(path, 'log.json')
    try:
        with open(json, 'r') as f:
            data = json.load(json)
            out = ", ".join([
                kwargs[w] if w in kwargs.keys() else data[w] \
                for w in columns]
            )
    except:
        continue

    csv.write(out + '\n')
    
csv.close()

