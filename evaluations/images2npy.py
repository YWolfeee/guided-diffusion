import os
import numpy as np
from PIL import Image
import blobfile as bf

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def read_images(data_dir, image_size=32):
    all_files = _list_image_files_recursively(data_dir)

    class_names = [bf.basename(path).split("_")[0] for path in all_files]
    sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    classes = [sorted_classes[x] for x in class_names]
    
    images = []
    for path in all_files:

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
        arr = np.array(pil_image.convert("RGB"))
        images.append(arr)

    return np.stack(images), np.array(classes)

def save_images(arr, classes, fn):
    np.savez(fn, arr, classes)

def label_filter(arr, classes, class_id):
    return arr[classes == class_id], classes[classes == class_id]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--positive_id', type=int, default=None)
    args = parser.parse_args()

    data_dir = './datasets/mnist_test'
    save_path = './evaluations/ref/' + data_dir.split('/')[-1]
    arr, labels = read_images(data_dir, 32)

    if args.positive_id is not None:
        arr, labels = label_filter(arr, labels, args.positive_id)
        save_path += f'_{args.positive_id}'

    save_images(arr, labels, save_path + '.npz')
