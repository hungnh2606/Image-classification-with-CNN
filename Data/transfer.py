from torch import Tensor
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights
import torch
import torch.nn as nn
from torchinfo import summary

# model = resnet50(weights=ResNet50_Weights.DEFAULT)
# model.fc = nn.Linear(2048,20)

model = vgg16(weights=VGG16_Weights.DEFAULT)

print(model.classifier[6])
model.classifier[6] = nn.Linear(4096,10)


# for name, param in model.named_parameters():
#     if "fc." not in name and "layer4." not in name:
#         param.requires_grad = False
#     print(name,param.requires_grad)
#
summary(model, (1,3,224,224),device="cpu")

