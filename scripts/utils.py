import torch
from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image

to_pil = T.ToPILImage()


def save_example(gen, val_loader, epoch, folder, device):
    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)

    gen.eval()
    with torch.no_grad():
        y_preds: torch.Tensor = gen(x)
        y_fakes = torch.chunk(y_preds.squeeze(), 4, dim=0)
        y_fakes = torch.cat(y_fakes, dim=2)
        y_reals = torch.chunk(y.squeeze(), 4, dim=0)
        y_reals = torch.cat(y_reals, dim=2)
        to_pil(x.squeeze()).save(folder + f"/input_{epoch}.png")
        to_pil(y_fakes).save(folder + f"/y_gen_{epoch}.png")
        to_pil(y_reals).save(folder + f"/y_{epoch}.png")
        # if epoch == 1:
        #     y_reals = torch.chunk(y.squeeze(), 4, dim=0)
        #     for y_real, y_real_map in zip(y_reals, ["normal", "diffuse", "roughness", "specular"]):
        #         save_image(y_real * 0.5 + 0.5, folder + f"/label_{epoch}_{y_real_map}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint,
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
