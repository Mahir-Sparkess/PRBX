import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import PILToTensor, ToPILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


both_transform = A.Compose(
    [A.Resize(width=256, height=256)], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)


class FabricDataset(Dataset):
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

        input_image = np.array(Image.open(filepath))
        target_images = [
            np.array(Image.open(os.path.join(self.target_dir, f"{filename.split('.')[0]}-outputs-{i}-.png")))
            for i in range(4)
        ]

        input_tensor = transform_only_input(image=input_image)["image"]
        target_tensors = [transform_only_mask(image=target_image)["image"] for target_image in target_images]

        target_tensor = torch.cat(target_tensors[:], dim=0)
        return input_tensor, target_tensor


def test():
    dataset = FabricDataset(dataset_dir=os.path.join(os.getcwd(), "Dataset"))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    x, y = next(iter(dataloader))
    print(x.shape)
    print(y.shape)
    from scripts.render import get_rendered_material
    rendered_img = get_rendered_material(y.squeeze())
    print(rendered_img.shape)
    # img = ToPILImage()(rendered_img)
    # img.show()


if __name__ == "__main__":
    test()
