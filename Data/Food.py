from typing import Tuple, List, Dict

import requests
import zipfile
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision import datasets
import os
import pathlib

from PIL import Image
from torchvision import transforms
# Setup path to data folder
data_path = Path("D:\Code Pytorch\Data\Food")
image_path = data_path / "pizza_steak_sushi"

def get_data():
    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")

    image_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak, sushi data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)


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



def plot_image():
    random.seed(42)  # <- try changing this and see what happens

    # 1. Get all image paths (* means "any combination")
    image_path_list = list(image_path.glob("*/*/*.jpg"))

    # 2. Get random image path
    random_image_path = random.choice(image_path_list)

    # 3. Get image class from path name (the image class is the name of the directory where the image is stored)
    image_class = random_image_path.parent.stem

    # 4. Open image
    img = Image.open(random_image_path)

    # 5. Print metadata
    print(f"Random image path: {random_image_path}")
    print(f"Image class: {image_class}")
    print(f"Image height: {img.height}")
    print(f"Image width: {img.width}")
    img
    img_as_array = np.asarray(img)

    # Plot the image with matplotlib
    plt.figure(figsize=(10, 7))
    plt.imshow(img_as_array)
    plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
    plt.axis(False);



def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

def image_folder():
    data_transform = transforms.Compose([
        # Resize the images to 64x64
        transforms.Resize(size=(64, 64)),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5),  # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor()  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
    ])
    train_data = datasets.ImageFolder(root=train_dir,  # target folder of images
                                      transform=data_transform,  # transforms to perform on data (images)
                                      target_transform=None)  # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=data_transform)
    class_names = train_data.classes
    print(class_names)

    class_dict = train_data.class_to_idx
    print(class_dict)
    print(len(class_names),len(class_dict))

    print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=32,
                                  num_workers=4,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=32,
                                 num_workers=4,
                                 shuffle=False)
    train_dataloader,test_dataloader
    print(f"Train dataloader:{train_dataloader}")
    print(f"Train dataloader:{test_dataloader}")

def find_classes(directory: str) -> Tuple[List[str], Dict[str,int]]:
    # get class name
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    print(classes,class_to_idx)
    return classes, class_to_idx

class ImageFolderCustom(Dataset):

    def __init__(self, targ_dir: str, transform=None):

        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))

        self.transform = transforms

        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
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
    # # Setup train and testing paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    print(train_dir, test_dir)

    # find_classes(train_dir)
    train_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])

    train_data_custom = ImageFolderCustom(targ_dir=train_dir,
                                          transform=train_transform)
    test_data_custom = ImageFolderCustom(targ_dir=test_dir,
                                         transform=test_transform)

    print(train_data_custom,test_data_custom)
    print(len(train_data_custom),len(test_data_custom))
    print(train_data_custom.classes,train_data_custom.class_to_idx)
