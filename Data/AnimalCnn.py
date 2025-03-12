import os.path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize, Compose


class AnimalDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        if train:
            mode = "train"
        else:
            mode = "test"
        root = os.path.join(root, mode)
        self.categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
        self.transform = transform

        self.images_path = []
        self.labels = []

        for i, category in enumerate(self.categories):
            data_files_path = os.path.join(root, category)
            for file_name in os.listdir(data_files_path):
                file_path = os.path.join(data_files_path, file_name)
                #print(file_path)
                self.images_path.append(file_path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

if __name__ == '__main__':
    root = "D:\Code Pytorch\Data\Animal"
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
    dataset = AnimalDataset(root=root, train=True, transform=transform)
    training_data_loader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        drop_last=False
    )

    for images, labels in training_data_loader:
        print(images.shape)
        print(labels)


