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

    def __init__(self, image_dir, transform=None, head=False, hands=False, silhouette_dir=None):
        super(ImageDataset, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.image_dir = image_dir
        self.silhouette_dir = silhouette_dir
        self.transform = transform
        self.image_paths = os.listdir(self.image_dir)
        self.head = head
        self.hands = hands

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).to(device=self.device)
        landmarks = lm.Landmarks(image, head=self.head, hands=self.hands)
        height, width, channels = image.shape
        image_size = [height, width]
        if self.silhouette_dir != None:
            silhouette_path = os.path.join(self.silhouette_dir, self.image_paths[index])
            silhouette = cv2.imread(silhouette_path)
            # silhouette = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            silhouette = cv2.bitwise_not(cv2.cvtColor(silhouette, cv2.COLOR_BGR2RGB))
            silhouette = torch.from_numpy(silhouette).to(device=self.device)
            return (image, landmarks, image_size, silhouette)
        else:
            return (image, landmarks, image_size)

