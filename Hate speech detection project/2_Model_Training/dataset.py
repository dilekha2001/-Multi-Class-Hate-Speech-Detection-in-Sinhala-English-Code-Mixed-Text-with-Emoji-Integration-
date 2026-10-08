"""
dataset.py

PyTorch Dataset wrapper around the processed CSVs, used identically
by the baseline (no-emoji) and proposed (with-emoji) training runs.
"""

import torch
from torch.utils.data import Dataset

LABEL2ID = {"neutral": 0, "offensive": 1, "hate": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = [LABEL2ID[l] for l in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
