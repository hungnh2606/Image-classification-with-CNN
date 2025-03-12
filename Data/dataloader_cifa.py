from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from  Cifadataset import MyDataset
import numpy as np


if __name__ == '__main__':
    #training_data = CIFAR10(root="Data/cifa-10-data/cifar-10-batches-py", train=True)
    training_data = MyDataset(root='D:\Code Pytorch\Data\cifa-10-data\cifar-10-batches-py',train=True, transform= ToTensor())
    training_data_loader = DataLoader(
        dataset = training_data,
        batch_size = 16,
        num_workers= 4,
        shuffle= True,
        drop_last= False,
    )

    for images, labels in training_data_loader:
        print(images.shape)
        print(labels)
