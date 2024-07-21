from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from xrkit.base import CONFIG

# mypy: disable-error-code="valid-type, arg-type"


class RSUASegmentationDataset(Dataset):
    def __init__(self, data_subset: str) -> None:
        """
        Initialize the SegmentationDataset.

        Parameters:
            data_subset (str):
                Subset of the data to load, either 'train' or 'test'.
        """

        self.data_subset = data_subset
        self.validation_percentage = 0.2

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((CONFIG.base.image_size, CONFIG.base.image_size)),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.set = self.split_data()

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""

        return len(self.set)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.set[index]
        image = Image.fromarray(np.squeeze(np.load(image_path) * 255, axis=2).astype("uint8"))

        mask_path = Path(
            image_path.parents[0].with_stem("npy_masks"), image_path.name.replace("Images", "Mask")
        )
        mask = Image.fromarray(np.squeeze(np.load(mask_path) * 255, axis=2).astype("uint8"))

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask

    def split_data(self) -> pd.DataFrame:
        """Splits the dataset into training and testing subsets."""

        np.random.seed(34)

        test_files = pd.read_excel(
            Path(CONFIG.data.rsua_segmentation.raw_path, "Split_Data_RSUA_Paths.xlsx"), sheet_name="test"
        )["images_path"].values

        leftover_files = [
            file
            for file in Path(CONFIG.data.rsua_segmentation.raw_path).rglob("*.npy")
            if "Images" in file.stem and file not in test_files
        ]
        test_files = [Path(CONFIG.data.raw.path, file) for file in test_files]

        n_validation = int(len(leftover_files) * self.validation_percentage)
        validation_files = np.random.choice(leftover_files, n_validation)

        train_files = [file for file in leftover_files if file not in validation_files]

        data_mapping = {
            "train": train_files,
            "validation": validation_files,
            "test": test_files,
        }

        if self.data_subset in data_mapping:
            return data_mapping[self.data_subset]
        else:
            raise ValueError("Invalid data type. Choose from 'train', 'validation' or 'test'.")


if __name__ == "__main__":
    dataset = RSUASegmentationDataset(data_subset="test")
    dataset.__getitem__(0)
