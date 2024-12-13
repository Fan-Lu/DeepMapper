
import numpy as np
import math
import os
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 as cv
from torch import optim

class WoundDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the root directory containing 'DA_Cropped' and 'DB-Uncropped'.
            transform (callable, optional): Optional transforms to be applied to both inputs and labels.
        """
        self.root_dir = root_dir
        self.cropped_dir = os.path.join(root_dir, "Davinci_Processed_Copy")
        self.uncropped_dir = os.path.join(root_dir, "DB-749NG_UCSC")
        self.transform = transform

        # Collect all cropped and uncropped file paths
        self.cropped_paths = glob(os.path.join(self.cropped_dir, "**/*.JPG"), recursive=True)
        self.uncropped_paths = glob(os.path.join(self.uncropped_dir, "**/*.JPG"), recursive=True)

        # Create a dictionary for cropped paths for quick lookup
        self.cropped_dict = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in self.cropped_paths
        }
        self.cropped_dict_new = {}
        for key, val in self.cropped_dict.items():
            ss = key.split('_')
            if len(ss) == 2:
                self.cropped_dict_new[ss[0]] = val
        # Filter uncropped paths that have a matching cropped image
        self.data_pairs = [
            (path, self.cropped_dict.get(os.path.splitext(os.path.basename(path))[0]))
            for path in self.uncropped_paths
            if os.path.splitext(os.path.basename(path))[0] in self.cropped_dict_new
        ]

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        uncropped_path, cropped_path = self.data_pairs[idx]

        # Load images
        # uncropped_image = Image.open(uncropped_path).convert("RGB")
        # cropped_image = Image.open(cropped_path).convert("RGB")

        uncropped_image = cv.imread(uncropped_path)[600:3500, 2200:5000]
        cropped_image = cv.imread(cropped_path)[600:3500, 2200:5000]
        cropped_image = np.where(cropped_image > 0, 1, cropped_image)
        cropped_image = (cropped_image[:, :, 0] & cropped_image[:, :, 1] & cropped_image[:, :, 2]) * 255

        cropped_image = Image.fromarray(cropped_image)
        cropped_image = cropped_image.resize((256, 256))
        uncropped_image = Image.fromarray(cv.cvtColor(uncropped_image, cv.COLOR_BGR2RGB)).convert("RGB")
        uncropped_image = uncropped_image.resize((256, 256))

        # Apply transforms if needed
        if self.transform:
            uncropped_image = self.transform(uncropped_image)
            cropped_image = self.transform(cropped_image)
        imgs = {"image": uncropped_image, "mask": cropped_image}

        return imgs