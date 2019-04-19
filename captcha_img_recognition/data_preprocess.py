import os
from PIL import Image

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import one_hot_encoding as ohe
import setting

class MyData(Dataset):
    def __init__(self, folder, transform=None):
        self.train_img_file_paths = [os.path.join(folder, img) for img in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        return len(self.train_img_file_paths)

    def __getitem__(self, idx):
        img_path = self.train_img_file_paths[idx]
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        label = ohe.encode(img_name[:4])
        return img, label


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

def get_train_dataloader():
    dataset = MyData(setting.train_img_path, transform=transform)
    return DataLoader(dataset, batch_size=10, shuffle=True)

def get_valid_dataloader():
    dataset = MyData(setting.valid_img_path, transform=transform)
    return DataLoader(dataset, batch_size=2, shuffle=True)

def get_test_dataloader():
    dataset = MyData(setting.test_img_path, transform=transform)
    return DataLoader(dataset, batch_size=5, shuffle=True)