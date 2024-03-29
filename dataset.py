import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from mask_creator import MaskCreator

class SeaSpeciesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.mask_creator = MaskCreator()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        mask, class_name = self.mask_creator.__createMask__(self.images[index])

        if class_name == "furcullaria":
            mask[mask == 1.0] = 2.0
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask