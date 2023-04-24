import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset import FabricDataset
from generator import Generator
from scripts import utils
from scripts.render import get_rendered_material as render

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
g = Generator(in_channels=3)
CHECKPOINT_GEN = "D:/PBRX/PRBX/saved weights/gen2.pth.tar"
DATASET_DIR = "D:/PBRX/PRBX/Dataset"
utils.load_model(CHECKPOINT_GEN, g, DEVICE)
fabric_dataset = FabricDataset(DATASET_DIR)
dataloader = DataLoader(fabric_dataset, batch_size=1, shuffle=True)
x, _ = next(iter(dataloader))
g.eval()
y = g(x)
g.train()


if __name__ == "__main__":
    to_pil = T.ToPILImage()

    to_pil(render((y * 0.5 + 0.5).squeeze())).show()

    y_fakes = torch.chunk((y * 0.5 + 0.5).squeeze(), 4, dim=0)
    y_fakes = torch.cat(y_fakes, dim=2)

    to_pil(y_fakes.squeeze()).show()
    to_pil((x * 0.5 + 0.5).squeeze()).show()
    exit(0)

