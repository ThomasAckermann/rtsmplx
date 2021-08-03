import os
import torch
from torch.utils.data import Dataset
import cv2
import rtsmplx.landmarks as lm


class ImageDataset(Dataset):
    """image dataset

    Keyword arguments:
    image_dir     --  string of the directory of the images
    transform   --  transforms that are applied to the images (default: None)
    """

    def __init__(self, image_dir, transform=None, head=False, hands=False):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = os.listdir(self.image_dir)
        self.head = head
        self.hands = hands

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        image = torch.from_numpy(cv2.imread(image_path))# .type(torch.int8)
        landmarks = lm.Landmarks(image, head=self.head, hands=self.hands)
        return (image, landmarks)
