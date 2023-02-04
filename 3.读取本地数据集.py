# 读取本地数据，读取注释过的数据分类，图片dataset

import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
# Creating a Custom Dataset for your files
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

# 基于dataset，把图片按照不同标签读入
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)  # 随机旋转图像，增强数据
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# 通过文件名和label一一对应，得到训练集的data和测试集data
training_data = CustomImageDataset(annotations_file="./excel/annotations_cat_car.csv"
                                   ,img_dir="./images")

test_data = CustomImageDataset(annotations_file="./excel/annotations_cat_car_test.csv"
                               ,img_dir="./images_test")

# dataloader是一个模块，一个类，最开始导入的，实例化这个类，得到一个实例化对象
# minibatch 的方法，shuffle:打乱数据顺序；测试集分2批
train_dataloader = DataLoader(training_data, batch_size=6, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=2, shuffle=True)


# Display image and label.
train_features, train_labels = next(iter(train_dataloader))

# 随机会展示一个图片
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
img = img.swapaxes(0,2)
plt.imshow(img)
plt.show()
print(f"Label: {label}")

# 对类别用字典编码，0表示cat，1表示car
labels_map = {
    0: "Cat",
    1: "Car",
}

# 展示图片和标签，随机展示九张，九宫格，包含图像和类别的标签
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    img = img.swapaxes(0, 2)
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()







