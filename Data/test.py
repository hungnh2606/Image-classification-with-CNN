# import os
# import pickle
# path = "D:\Code Pytorch\Data\cifa-10-data\cifar-10-batches-py/test_batch"
#
# if os.path.exists(path):
#     with open(path, "rb") as f:
#         dict = pickle.load(f, encoding='bytes')
#         print(dict[b'data'])
#         print(dict[b'labels'])
#         print(type(dict))
#         print(dict.keys())
#         input()
# else:
#     print(f"Lỗi: Không tìm thấy tệp {path}")

# import os
#
# root = "D:\Code Pytorch\Data\cifa-10-data\cifar-10-batches-py"  # Hoặc giá trị root của bạn
# path = os.path.join(root, "test_batch")
#
# print("Root path:", root)
# print("Joined path:", path)
# print("Normalized path:", os.path.normpath(path))  # Chuẩn hóa đường dẫn
#
# if os.path.exists(path):
#     print("✅ Tệp tồn tại:", path)
# else:
#     print("❌ Không tìm thấy tệp:", path)

#
# from torchvision.datasets import ImageFolder
#
# train_dataset = ImageFolder(root="D:\Code Pytorch\Data\Animal\train\cat")
# index = 200
#
# image, label = train_dataset.__getitem__(index)
# image.show()
# print(image.size)
# print(label)
# print(train_dataset.classes)
# print(train_dataset.class_to_idx)

# import torch
# print(torch.__version__)  # Kiểm tra phiên bản PyTorch
# print(torch.cuda.is_available())  # Kiểm tra CUDA có hoạt động không
# print(torch.version.cuda)  # Kiểm tra phiên bản CUDA PyTorch hỗ trợ
# print(torch.backends.cudnn.version())  # Kiểm tra phiên bản cuDNN

import torch
print(torch.cuda.is_available())  # Nếu True là OK
print(torch.cuda.get_device_name(0))  # Kiểm tra tên GPU