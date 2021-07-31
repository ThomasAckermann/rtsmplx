import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import rtsmplx.landmarks as lm


class ImageDataset(Dataset):
    """ image dataset

    Keyword arguments:
    img_dir     --  string of the directory of the images 
    transform   --  transforms that are applied to the images (default: None) 
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = os.listdir(self.img_dir)
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_paths[index])
        image = read_image(img_path)
        landmarks = lm.Landmarks(image)
        return image, landmarks



