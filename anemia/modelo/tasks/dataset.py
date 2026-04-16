import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class ConjuntivaDataset(Dataset):
    def __init__(self, x_data, y_data, in_chans=4):
        self.x_data = x_data
        self.y_data = y_data
        self.in_chans = in_chans

        mean = [0.5] * in_chans
        std = [0.5] * in_chans

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        image = self.x_data[idx]
        label = self.y_data[idx]

        # Asegurarse de que tenga 4 canales
        if image.shape[-1] == 3 and self.in_chans == 4:
            alpha = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255
            image = np.concatenate((image, alpha), axis=-1)

        image = self.transform(image)
        return image, torch.tensor(label).long()
