from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from preprocess import ben_graham_preprocess


class DRDataset(Dataset):
    def __init__(self, csv_path, train=False):
        self.df = pd.read_csv(csv_path)
        self.train = train

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.eval_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(row["image_path"])
        label = int(row["diagnosis"])

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        image = ben_graham_preprocess(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train:
            image = self.train_transform(image)
        else:
            image = self.eval_transform(image)

        return image, torch.tensor(label, dtype=torch.long)