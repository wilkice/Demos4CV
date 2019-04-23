import os

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import one_hot_encoding as ohe
import setting
import cv2


# TODO： use opencv, need to close?

class MyData(Dataset):
    """A customized data loader for captcha images"""
    def __init__(self, folder, transform=None, preload=False):
        """Initialize the captcha dataset
        
        Args:
            - folder: relative root directory of images
            - transform: a custom transform function
            - preload: if preload the dataset to memory
        TODO:transform or transforms in args
        """
        self.images = None
        self.labels = None
        self.filenames = []
        self.folder = folder
        self.transform = transform

        # get relative filenames 
        self.filenames = [os.path.join(folder, img) for img in os.listdir(folder)]

        # if preload dataset to memory
        if preload:
            self._preload()

        self.length = len(self.filenames)

    def _preload(self):
        """preload dataset to momory"""
        self.labels = []
        self.images = []
        for image_fn in self.filenames:
            image = cv2.imread(image_fn)
            self.images.append(image.copy())
            label_string = image_fn[-8:-4]
            label=ohe.encode(label_string)
            self.labels.append(label)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        if self.images:
            # if preload
            image = self.images[idx]
            label = self.labels[idx]
        else:
            image_name = self.filenames[idx]
            image = cv2.imread(image_name)
            label_string = image_name[-8:-4]
            label=ohe.encode(label_string)
        # if we have transform functions
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.length

# convert numpy.ndarray (H x W x C) in the range [0, 255]
# to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
transform = transforms.Compose([
    transforms.ToTensor()
])

def get_train_dataloader():
    dataset = MyData(setting.train_img_path, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True,num_workers=2)

def get_valid_dataloader():
    dataset = MyData(setting.valid_img_path, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

def get_test_dataloader():
    dataset = MyData(setting.test_img_path, transform=transform)
    return DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
