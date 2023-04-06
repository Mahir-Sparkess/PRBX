import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import PILToTensor, ToPILImage


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

        input_image = convert_tensor(Image.open(filepath))
        target_images = [
            convert_tensor(Image.open(os.path.join(self.target_dir, f"{filename.split('.')[0]}-outputs-{i}-.png")))
            for i in range(4)]
        target_image = torch.cat(target_images[:], dim=0)
        return input_image, target_image


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
