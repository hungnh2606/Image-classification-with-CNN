import torch
import torch.nn as nn
from torch.ao.nn.quantized import BatchNorm2d


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, num_class= None):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer_1 = nn.Sequential(
            nn.Linear(in_features=3 * 32 * 32, out_features=256),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU()
        )
        self.layer_3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU()
        )
        self.layer_4 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU()
        )
        self.layer_5 = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_class),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.conv1 = self._make_block(3,8)
        self.conv2 = self._make_block(8,16)
        self.conv3 = self._make_block(16,32)
        self.conv4 = self._make_block(32,64)
        self.conv5 = self._make_block(64,128)

        self.fc1 = self._make_fc(6272,512)
        self.fc2 = self._make_fc(512,1024)
        self.fc3 = self._make_fc(1024,num_classes)


    def _make_fc(self,in_feature,out_feature):
        return nn.Sequential(
            nn.Linear(in_features=in_feature,out_features=out_feature),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )


    def _make_block(self,in_channels,out_channels,kernel_size=None):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    model = SimpleCNN()
    input_data = torch.rand(8,3,224,224)
    result = model(input_data)
    print(result.shape)