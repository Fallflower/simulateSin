from torch.utils.data import dataset, dataloader
import torch
import numpy as np


class genDataSet(dataset.Dataset):
    def __init__(self, size, target_f=torch.sin):
        self.x = torch.rand(size) * 2 * torch.tensor(np.pi)  # [0, 2pi)
        self.y = target_f(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def get_dataloader(size, mode="train"):
    ds = genDataSet(size)
    if mode == "train":
        return dataloader.DataLoader(ds, batch_size=64, shuffle=True)
    else:
        return dataloader.DataLoader(ds, batch_size=64, shuffle=False)