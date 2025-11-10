#
# Copyright (c) 2025 Your Name or Organization
# Portions Copyright (C) 2019-2023 Maxim Integrated Products, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
"""
Apple Spectral Dataset for AI8X Training
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import ai8x

class AppleSpectraDataset(Dataset):
    """ Custom dataset for spectral data from apples """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = {
            "CRIPPSRED": 0,
            "GALA": 1,
            "JULIETTE": 2,
            "STORY": 3
        }

        # Load dataset
        for variety, label in self.classes.items():
            variety_path = os.path.join(self.root_dir, variety) + "/"
            if os.path.exists(variety_path):
                for file in os.listdir(variety_path):
                    if file.endswith(".txt"):
                        spectrum = np.loadtxt(os.path.join(variety_path, file)) 
                        self.data.append(spectrum)
                        self.labels.append(label)

        # Convert to tensors
        self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(1)  # Add channel dim
        self.labels = torch.tensor(self.labels, dtype=torch.long)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx], self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def apple_spectra_get_datasets(data, load_train=True, load_test=True):
    """
    Load the Apple Spectral dataset.
    """
    (data_dir, args) = data

    # Define transformations (optional)
    transform = ai8x.normalize(args=args)

    test_data_dir = data_dir + "/Test/"
    train_data_dir = data_dir + "/Train/"

    if load_train:
        train_dataset = AppleSpectraDataset(root_dir=train_data_dir,
                                            transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = AppleSpectraDataset(root_dir=test_data_dir,
                                           transform=transform)
    else:
        test_dataset = None

    return train_dataset, test_dataset


# Register dataset in AI8X
datasets = [
    {
        'name': 'AppleSpectra',
        'input': (1, 288),  # 1D input (channels, features)
        'output': ('CRIPPSRED', 'GALA', 'JULIETTE', 'STORY'),
        'loader': apple_spectra_get_datasets,
    },
]

