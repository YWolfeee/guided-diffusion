from PIL import Image
import torch
import numpy as np
import torch.nn as nn

from guided_diffusion.celebA.model import Backbone

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.PReLU(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNetFace(nn.Module):
    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNetFace, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(512 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.bn5(x)

        return x

def resnet_face18(use_se=True, **kwargs):
    model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=use_se, **kwargs)
    return model

def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k[len('module.'):]: v for k, v in pretrained_dict.items() if k[len('module.'):] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def load_arcface(model_path, device):
    # model = resnet_face18(use_se=False)
    # model.eval()
    # load_model(model, model_path)
    # model.to(device)
    # return model
    model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    return model

def arcface_forward_path(model, path, device='cuda'):
    image = Image.open(path).convert('RGB')
    data = torch.tensor(np.array(image).transpose(2, 0, 1), device=device).unsqueeze(0) / 127.5 - 1
    output = arcface_forward(model, data)
    return output


def arcface_forward(model, data):
    # data = (data[:, 0, :, :] * 0.299 + data[:, 1, :, :] * 0.587 + data[:, 2, :, :] * 0.114).unsqueeze(1)
    # data = nn.functional.interpolate(data, size=128, mode='bilinear', align_corners=True)
    # output = model(data)
    # return output
    data = (data - 0.5) / 0.5   # normalize
    data = data[:, :, 35:223, 32:220]
    face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
    data = face_pool(data)
    feature = model(data)
    return feature

def cosine(feat1, feat2):
    return nn.functional.cosine_similarity(feat1, feat2)

if __name__ == '__main__':
    # with torch.no_grad():
    #     model = load_arcface('/home/linhw/code/guided-diffusion/guided_diffusion/celebA/resnet18_110.pth', 'cuda')    
    #     image = Image.open('/home/linhw/code/guided-diffusion/guided_diffusion/celebA/celeba_hq_256/00000.jpg').convert('L')
    #     image.save('tmp.png')
    #     image = Image.open('/home/linhw/code/guided-diffusion/guided_diffusion/celebA/celeba_hq_256/00000.jpg').convert('RGB')
    #     data = torch.tensor(np.array(image).transpose(2, 0, 1)).unsqueeze(0) / 127.5 - 1
    #     output = arcface_forward(model, data.cuda())
    #     # data = ((data[0] + 1) * 127.5).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)

    #     # Image.fromarray(data[:, :, 0]).save('tpm.png')
    #     print(output.shape)
    model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
    model.load_state_dict(torch.load('/home/linhw/code/guided-diffusion/ckpts/celebA/model_ir_se50.pth'))
    model.eval()
    face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))

    image1 = Image.open('/home/linhw/code/guided-diffusion/datasets/celeba_hq_256/00000.jpg').convert('RGB')
    data1 = torch.tensor(np.array(image1).transpose(2, 0, 1)).unsqueeze(0) / 127.5 - 1
    data1 = (data1 - 0.5) / 0.5
    data1 = face_pool(data1)
    feature1 = model(data1)

    image2 = Image.open('/home/linhw/code/guided-diffusion/datasets/celeba_hq_256/00001.jpg').convert('RGB')
    data2 = torch.tensor(np.array(image2).transpose(2, 0, 1)).unsqueeze(0) / 127.5 - 1
    data2 = (data2 - 0.5) / 0.5
    data2 = face_pool(data2)
    feature2 = model(data2)

    image3 = Image.open('/home/linhw/code/guided-diffusion/datasets/celeba_hq_256/29894.jpg').convert('RGB')
    data3 = torch.tensor(np.array(image3).transpose(2, 0, 1)).unsqueeze(0) / 127.5 - 1
    data3 = (data3 - 0.5) / 0.5
    data3 = face_pool(data3)
    feature3 = model(data3)

    print(nn.functional.cosine_similarity(feature1, feature2))
    print(nn.functional.cosine_similarity(feature1, feature3))
    print(nn.functional.cosine_similarity(feature3, feature2))