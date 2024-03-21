import matplotlib.pyplot as plt
import imageio
import os
import re
import wandb
from PIL import Image
base_dir = '/home/linhw/code/guided-diffusion/temp/celebA-present'
baselines = [
    'manifold+iter=5',
    # 'manifold+iter=1',
    'dynamic-two-0.1-a+iter=5',
    # 'dynamic-two-0.1-a+iter=1'
]
settings = [
    '00000+00000+00000',
    '00000+00001+00002',
    '15148+27812+16239',
    '15148+15148+15148'
]

def combine_images(img_path1, img_path2, output_path='combined_image.jpg'):
    # Open the images
    img2 = Image.open(img_path2)
    img1 = Image.open(img_path1)
    img1 = img1.resize((img2.width, img2.height))

    # Calculate dimensions for the combined image
    total_width = img1.width + img2.width

    max_height = max(img1.height, img2.height)

    # Create a new blank image with the calculated dimensions
    new_img = Image.new('RGB', (total_width, max_height))

    # Paste the first image on the left
    new_img.paste(img1, (0, (max_height - img1.height) // 2))

    # Paste the second image on the right
    new_img.paste(img2, (img1.width, (max_height - img2.height) // 2))

    # Save the combined image
    new_img.save(output_path)
    return output_path


def is_float(value):
    pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
    return bool(pattern.match(value))

def parse_log(log):
    values = []
    with open(log, 'r') as f:
        lines = [x.strip() for x in f.readlines()]
    for line in lines:
        if is_float(line):
            values.append(float(line))
    return values

def parse_dir_name(dir_name, baseline, setting):
    images = []
    losses = []

    for dir_name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, dir_name, os.listdir(os.path.join(base_dir, dir_name))[0])
        if baseline not in full_path or setting not in full_path:
            continue
        img = os.path.join(full_path, 'samples_16x256x256x3.png')
        log = os.path.join(full_path, 'log.txt')
        if not os.path.exists(img) or not os.path.exists(log):
            print(dir_name)
            continue

        dir_name = dir_name.split("+")

        values = -parse_log(log)[-1]

        images.append(img)
        losses.append(values)
    
    return images, losses

def plot_loss_curve(loss_values, file_path='temp'):
    
    filenames = []
    
    for i in range(1, len(loss_values) + 1):
        # Plot the loss curve up to the current step
        plt.figure(figsize=(10, 10))
        plt.plot(range(1, i + 1), loss_values[:i], marker='o', color='b')
        plt.title('Loss Curve')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.xlim(0, len(loss_values)+1)
        plt.ylim(min(loss_values) - 0.5 * min(loss_values), max(loss_values) + 0.1 * max(loss_values))
        plt.grid(True)
        
        # Save the plot as a temporary file
        filename = f'{file_path}/frame_{i}.png'
        plt.savefig(filename)
        plt.close()
        filenames.append(filename)
    return filenames

def create_gif(gif_path, fps, filenames):
    # Compile all the images into a GIF
    with imageio.get_writer(gif_path, mode='I', duration=1000/fps) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    

if __name__ == '__main__':

    for baseline in baselines:
        for setting in settings:
            image_path, losses = parse_dir_name(base_dir, baseline, setting)
            image_path, losses = zip(*list(sorted(zip(image_path, losses), key=lambda x: -x[1])))
            file_path = f'./temp/{baseline}_{setting}'
            os.makedirs(file_path, exist_ok=True)
            loss_path = plot_loss_curve(losses, file_path=file_path)
            outpath = os.path.join(file_path, 'gif')
            os.makedirs(outpath, exist_ok=True)
            for i, (image, loss) in enumerate(zip(image_path, loss_path)):
                op = combine_images(image, loss, os.path.join(outpath, f'combined_{i}.jpg'))
            
            create_gif(f'{outpath}/{baseline}_{setting}.gif', 10, [os.path.join(outpath, f'combined_{i}.jpg') for i in range(len(image_path))])
