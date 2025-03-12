from pathlib import Path

from jupyterlab.extensions import entry
from torchvision import datasets, transforms
import os
import pathlib

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List
# Setup path to data folder
data_path = Path("D:\Code Pytorch\Data")
image_path = data_path / "Animal"

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str or pathlib.Path): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
# Setup train and testing paths

def image_folder():
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    print(train_dir, test_dir)
    # Write transform for image
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor()  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
    ])

    train_data = datasets.ImageFolder(root=train_dir,  # target folder of images
                                      transform=data_transform,  # transforms to perform on data (images)
                                      target_transform=None)  # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=data_transform)

    print(f"Train data:\n{train_data}\nTest data:\n{test_data}")



def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"No classes found in {directory}")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    print(f"class names:\n{classes}")
    print(f"class_to_idx:\n{class_to_idx}")
    return classes, class_to_idx
class AnimalDataset(Dataset):

    def __init__(self, targ_dir: str, transform=None):
        self.paths = list(pathlib.Path(targ_dir).glob("*/*"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

if __name__ == "__main__":
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Augment train data
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    # Don't augment test data, only reshape
    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    train_data_custom = AnimalDataset(targ_dir=train_dir,
                                      transform=train_transforms)
    test_data_custom = AnimalDataset(targ_dir=test_dir,
                                     transform=test_transforms)
    print(len(train_data_custom),len(test_data_custom))
    print(train_data_custom.classes,train_data_custom.class_to_idx)
    print(f"train data:\n{train_data_custom}")
    print(f"test data:\n{test_data_custom}")

