import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor


class CustomDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.input_dir = os.path.join(dataset_dir, "Input")
        self.target_dir = os.path.join(dataset_dir, "Target")
        self.images = os.listdir(self.input_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        filepath = os.path.join(self.input_dir, filename)

        convert_tensor = PILToTensor()

        input_image = convert_tensor(Image.open(filepath))
        target_images = [
            convert_tensor(Image.open(os.path.join(self.target_dir, f"{filename.split('.')[0]}-outputs-{i}-.png")))
            for i in range(4)]
        target_image = torch.cat(target_images, dim=2)
        return input_image, target_image
