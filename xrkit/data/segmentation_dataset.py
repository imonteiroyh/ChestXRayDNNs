from pathlib import Path

import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import Tuple
import torch
from xrkit.base import CONFIG


class SegmentationDataset(Dataset):
    def __init__(self, data_subset: str) -> None:
        """
        Initialize the SegmentationDataset.

        Parameters:
            data_subset (str):
                Subset of the data to load, either 'train' or 'test'.
        """
        self.data_subset = data_subset

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((CONFIG.base.image_size, CONFIG.base.image_size)),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.set_info = self.split_data()

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        
        return len(self.set_info)

    def __getitem__(self, index: int, transform: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset.

        Parameters:
            index (int):
                Index of the sample to retrieve.
            transform (bool):
                Whether to apply transformations to the data. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Tuple containing the image and its corresponding mask.
        """

        image_path = self.info.iloc[index]["Image Index"]
        top_left_x, top_left_y, width, height = (
            self.info.iloc[index][["Bbox [x", "y", "w", "h]"]].astype(int).values
        )
        image = Image.open(next(Path(CONFIG.data.raw.path).rglob(image_path)).as_posix()).convert("L")

        image_shape = image.size[::-1]
        mask = np.zeros(image_shape, dtype=np.uint8)
        mask[top_left_y : top_left_y + height, top_left_x : top_left_x + width] = 255.0
        mask = Image.fromarray(mask).convert("L")

        if transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def split_data(self) -> pd.DataFrame:
        """Splits the dataset into training and testing subsets."""

        self.info = pd.read_csv(Path(CONFIG.data.raw.path, "BBox_List_2017.csv"))

        X = self.info["Image Index"]
        y = self.info["Finding Label"]

        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=CONFIG.base.seed
        )
        train_subset = self.info[self.info["Image Index"].isin(X_train.tolist())]
        test_subset = self.info[self.info["Image Index"].isin(X_test.tolist())]

        data_mapping = {"train": train_subset, "test": test_subset}

        if self.data_subset in data_mapping:
            return data_mapping[self.data_subset]
        else:
            raise ValueError("Invalid data type. Choose from 'train' or 'test'.")
