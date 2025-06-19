import re

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from visualize import visual_jpeg


class preprocess_dataset:
    def __init__(self, csv_dirs, meta_data):
        self.csv_dirs = csv_dirs
        self.meta_data = meta_data

    def read_csv(self):
        gt_data = {}
        for meta in self.meta_data.values:
            splited_meta = meta[0].split("\t")
            gt_data[splited_meta[0]] = splited_meta[6]

        self.EMG, self.AGE = [], []
        for csv_dir in tqdm(self.csv_dirs):
            emgs_df = pd.read_csv(csv_dir)
            PL_emg = emgs_df["PL"].values
            GM_emg = emgs_df["GM"].values
            GL_emg = emgs_df["GL"].values
            SO_emg = emgs_df["SO"].values

            match = re.search(r"ID\d{4}", csv_dir)

            if match:
                participant_id = match.group()  # 'ID0001'

                if participant_id in gt_data:
                    age = gt_data[participant_id]
            else:
                print("No ID found in path")

            self.EMG.append(PL_emg)
            self.AGE.append(age)

            self.EMG.append(GM_emg)
            self.AGE.append(age)

            self.EMG.append(GL_emg)
            self.AGE.append(age)

            self.EMG.append(SO_emg)
            self.AGE.append(age)

        # visual_jpeg(self.EMG, self.AGE)

    def split_dataset(self):
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(
            self.EMG, self.AGE, test_size=0.4, random_state=42  # 40%를 valid+test로 둠
        )

        # 2. 그 다음 temp를 다시 valid vs test로 나누기 (0.5:0.5 → 20:20)
        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

    def load_dataset(self, batch_size):
        train_dataset = EMGDataset(self.X_train, self.y_train)
        valid_dataset = EMGDataset(self.X_valid, self.y_valid)
        test_dataset = EMGDataset(self.X_test, self.y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        return [train_loader, valid_loader, test_loader]


class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.y = torch.tensor([float(val) for val in y], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_fn(batch):
    sequences, targets = zip(*batch)
    padded_X = pad_sequence(sequences, batch_first=True)  # (batch, max_len)
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    return padded_X, targets
