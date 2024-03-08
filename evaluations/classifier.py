import numpy as np
import torch
import os
import argparse
from PIL import Image
from torchvision.models import resnet18
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import ToTensor
from guided_diffusion.image_datasets import load_data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='/home/linhw/code/improved-diffusion/datasets/mnist_train')
    parser.add_argument('--test_npz', type=str, default='/home/linhw/code/guided-diffusion/evaluations/ref/mnist_test.npz')
    args = parser.parse_args()
    return args


def get_model(ckpts=None, num_classes=10):
    model = resnet18(num_classes=num_classes, weights=None)
    if ckpts is not None:
        model.load_state_dict(torch.load(ckpts))
    return model

def get_dataloader(train_dir, batch_size, image_size, random_crop=True):
    num_workers = 32
    train_loader = load_data(
        data_dir=train_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=True,
        random_crop=random_crop,
        deterministic=False,
        num_workers=num_workers
    )

    return train_loader

@torch.no_grad()
def compute_accuracy(model_path_or_model, npz_path, label=None):

    if type(model_path_or_model) == str:
        model = get_model(model_path_or_model, num_classes=10).cuda()
    else:
        model = model_path_or_model
    
    model.eval()
    npz = np.load(npz_path)
    image = npz['arr_0'].transpose(0, 3, 1, 2) / 127.5 - 1
    data = torch.tensor(image).float()
    logits = model(data.cuda())
    preds = logits.argmax(1)

    if label is None:
        label = torch.tensor(npz['arr_1']).cuda()

    acc = (preds == label).float().mean()
    return acc.item()

def train_classifier():
    args = get_args()
    
    train_dir = args.train_dir

    ITER = 200000
    batch_size = 256

    model = get_model(num_classes=10).cuda()

    for p in model.parameters():
        p.requires_grad = True
    
    opt = SGD(model.parameters(), lr=5e-2, weight_decay=0.001, momentum=0.9)
    scheduler = MultiStepLR(opt, milestones=[20000, 40000], gamma=0.1)

    train_loader = get_dataloader(
        train_dir, batch_size, 32
    )
   
    for i, batch in enumerate(train_loader):

        if i >= ITER:
            break
            
        model.train()
        x = batch[0]
        y = batch[1]['y']

        x = x.cuda()
        y = y.cuda()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        scheduler.step()

        if i % 200 == 0:
            print(f'iter {i}, loss: {loss.item()}')

        if i % 500 == 0:
            acc = []
            with torch.no_grad():
                model.eval()
                acc = compute_accuracy(model, args.test_npz)
            print(f'acc: {acc}')

            ds = args.test_npz.split('/')[-1].strip('.npz')
            os.makedirs(f'./eval_cls/{ds}', exist_ok=True)
            torch.save(model.state_dict(), f'./eval_cls/{ds}/model_{i}_{acc}.pth')

if __name__ == '__main__':
    train_classifier()