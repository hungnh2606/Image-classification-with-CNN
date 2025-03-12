import cv2
import numpy as np
from torch.utils.data import Dataset
import os
import pickle


class MyDataset(Dataset):
    def __init__(self, root, train=True,transform=None):
        if train:
            data_files = [os.path.join(root,"data_batch_{}".format(i)) for i in range(1,6)]
        else:
            data_files = [os.path.join(root,"test_batch")]

        print(data_files)

        self.images = []
        self.labels = []
        try:
            for data_file in data_files:
                with open(data_file, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    self.images.extend(dict[b'data'])
                    self.labels.extend(dict[b'labels'])
            print(len(self.labels))
            print(len(self.images))
        except Exception as e:
            print(f"Có lỗi xảy ra: {e}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((3, 32, 32)).astype(np.float32)  # RGB
        label = self.labels[idx]
        return image / 255., label

if __name__ == "__main__":
    dataset = MyDataset(root="D:\Code Pytorch\Data\cifa-10-data\cifar-10-batches-py",train=True)
    image, labels = dataset.__getitem__(345)
    image = np.reshape(image,(3,32,32))
    print("before", image.shape)
    image = np.transpose(image, (1, 2, 0))
    print("after", image.shape)
    print(image.dtype)
    print(labels)
    cv2.imshow("image", cv2.resize(image, (320,320)))
    cv2.waitKey(0)