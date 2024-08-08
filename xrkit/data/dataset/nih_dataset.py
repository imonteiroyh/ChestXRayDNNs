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


class NIHDataset(Dataset):
    def __init__(self, data_subset: str) -> None:
        """
        Initialize the NIHDataset.

        Parameters:
            data_subset (str):
                Subset of the data to load, either 'train' or 'test'.
        """
        self.labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

        self.data_subset = data_subset
        self.validation_percentage = 0.25

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((CONFIG.base.image_size, CONFIG.base.image_size)),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.preprocess_targets()
        self.set = self.split_data()

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""

        return len(self.set)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file = self.set[index]
        image_path = list(Path(CONFIG.data.nih_classification.raw_path).rglob(file))[0]
        image = Image.open(image_path).convert('RGB')

        targets = self.reference_file[self.reference_file['Image Index'] == file].iloc[0][self.labels].values.astype(int)

        image = self.transform(image).float()
        targets = torch.from_numpy(targets).float()

        return image, targets

    def preprocess_targets(self):
        self.reference_file = pd.read_csv(Path(CONFIG.data.nih_classification.raw_path, "Data_Entry_2017.csv"))

        # Removing labels with less than 1000 samples (only Hernia)
        self.reference_file = self.reference_file[self.reference_file['Finding Labels'].str.count('Hernia') == 0]

        labels = []
        for _, file in self.reference_file.iterrows():
            current_labels = {}
            current_labels['Image Index'] = file['Image Index']

            for label in self.labels:
                current_labels[label] = 1 if label in file['Finding Labels'] else 0

            labels.append(current_labels)

        self.targets = pd.DataFrame(labels)
        self.targets[self.labels] = self.targets[self.labels].astype(int)

        self.reference_file = pd.merge(self.reference_file, self.targets).reset_index(drop=True)

    def split_data(self) -> pd.DataFrame:
        """Splits the dataset into training and testing subsets."""

        with open(CONFIG.data.nih_classification.test_samples, 'r') as file:
            test_samples = file.read().split('\n')
        test_samples = self.reference_file[self.reference_file['Image Index'].isin(test_samples)]['Image Index'].values.tolist()
        
        with open(CONFIG.data.nih_classification.train_validation_samples, 'r') as file:
            train_validation_samples = file.read().split('\n')
        train_validation_samples = self.reference_file[self.reference_file['Image Index'].isin(train_validation_samples)]['Image Index'].values.tolist()
        
        # 25% patients, 20% data to validation
        validation_patients = pd.Series(self.reference_file[~self.reference_file['Image Index'].isin(test_samples)]['Patient ID'].unique()).sample(frac=self.validation_percentage, random_state=CONFIG.base.seed).values
        validation_samples = self.reference_file[self.reference_file['Patient ID'].isin(validation_patients)]['Image Index'].values.tolist()
        train_samples = list(set(train_validation_samples) - set(validation_samples))

        data_mapping = {
            "train": train_samples,
            "validation": validation_samples,
            "test": test_samples,
        }

        if self.data_subset in data_mapping:
            return data_mapping[self.data_subset]
        else:
            raise ValueError("Invalid data type. Choose from 'train', 'validation' or 'test'.")


if __name__ == "__main__":
    dataset = NIHDataset(data_subset="train")
    dataset.__getitem__(0)
