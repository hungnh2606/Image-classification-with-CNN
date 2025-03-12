from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights
import torch
import torch.nn as nn
from torchinfo import summary

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
# print(model)
print(model.classifier[1])
# print(model.fc)
# model.fc = nn.Linear(2048,20)
# for name, param in model.named_parameters():
#     if "fc." not in name and "layer4." not in name:
#         param.requires_grad = False
#     print(name, param.requires_grad)

# summary(model, (2,3,224,224))

# for name, param in model.named_parameters():
#     if "fc." not in name and "layer4." not in name:
#         param.requires_grad = False
#     print(name,param.requires_grad)

#summary(model, (3,224,224))

# class MyResNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
#         del self.model.fc
#         self.fc1 = nn.Linear(2048,1024)
#         self.fc2 = nn.Linear(1024, num_classes)
#
#     def _forward_impl(self, x):
#         # See note [TorchScript super()]
#         x = self.model.conv1(x)
#         x = self.model.bn1(x)
#         x = self.model.relu(x)
#         x = self.model.maxpool(x)
#
#         x = self.model.layer1(x)
#         x = self.model.layer2(x)
#         x = self.model.layer3(x)
#         x = self.model.layer4(x)
#
#         x = self.model.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#
#         return x
#
#     def forward(self, x):
#         return self._forward_impl(x)
#
# if __name__ == "__main__":
#     model = MyResNet()
#     print(model)
#     image = torch.rand(2,3,224,224)
#     summary(model, input_size=(1, 3, 224, 224))
#     try:
#         output = model(image)
#         print(output.shape)
#     except Exception as e:
#         print(f"An error occurred: {e}")

