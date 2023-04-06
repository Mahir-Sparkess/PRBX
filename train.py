import torch
import numpy as np
import scripts.utils as utils
import torch.nn as nn
import torch.optim as optim
from dataset import FabricDataset
from generator import Generator
from discriminator import MapDiscriminator
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "D:/PBRX/PRBX/Dataset"
TRAIN_DIR = "Data/train"
VAL_DIR = "Data/val"
EVAL_DIR = "Evaluation"
LEARNING_RATE = 2e-3
BATCH_SIZE = 32
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"


def train(d, g, loader, opt_d, opt_g, l1, bce, g_s, d_s):
    loop = tqdm(loader, leave=True)

    for i, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.cuda.amp.autocast():
            y_fake = g(x)
            D_real = d(x, y)
            D_fake = d(x, y_fake.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
        d.zero_grad()
        d_s.scale(D_loss).backward()
        d_s.step(opt_d)
        d_s.update()

        with torch.cuda.amp.autocast():
            D_fake = d(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_g.zero_grad()
        g_s.scale(G_loss).backward()
        g_s.step(opt_g)
        g_s.update()


def evaluate(d, g, loader, l1, bce, g_err, d_err, epoch):
    d.eval()
    g.eval()
    d_losses = []
    g_losses = []
    loop = tqdm(loader, leave=True)
    for i, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.cuda.amp.autocast():
            g_x = g(x)
            d_g_x = d(x, g_x)
            d_y = d(x, y)
            g_loss = bce(d_g_x, torch.ones_like(d_g_x)) + (l1(g_x, y) * L1_LAMBDA)
            d_loss = (bce(d_y, torch.ones_like(d_y)) + bce(d_g_x, torch.zeros_like(d_g_x))) / 2
            g_losses.append(g_loss.cpu().detach().numpy())
            d_losses.append(d_loss.cpu().detach().numpy())
            break
    print(f"Epoch: {epoch} - G_Loss: {np.mean(g_losses):.2f} - D_Loss: {np.mean(d_losses):.2f}\n")
    d.train()
    g.train()
    g_err.append(np.mean(g_losses)), d_err.append(np.mean(d_losses))


def initialisation(load_checkpoints=False, save_model=False):
    d = MapDiscriminator(in_channels=3).to(DEVICE)
    g = Generator(in_channels=3).to(DEVICE)
    opt_d = optim.Adam(d.parameters(), lr=1e-6, betas=(0.5, 0.999))
    opt_g = optim.Adam(g.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    G_Losses = []
    D_Losses = []

    if load_checkpoints:
        # utils.load_checkpoint(CHECKPOINT_DISC, d, opt_d, LEARNING_RATE, DEVICE)
        utils.load_checkpoint(CHECKPOINT_GEN, g, opt_g, LEARNING_RATE, DEVICE)

    fabric_dataset = FabricDataset(DATASET_DIR)
    indices = list(range(len(fabric_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:10000], indices[10000:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(fabric_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(fabric_dataset, batch_size=1, sampler=val_sampler, num_workers=NUM_WORKERS)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train(d, g, train_loader, opt_d, opt_g, L1_LOSS, BCE, d_scaler, g_scaler)
        if save_model and epoch % 5 == 0:
            utils.save_checkpoint(g, opt_g, filename=CHECKPOINT_GEN)
            utils.save_checkpoint(d, opt_d, filename=CHECKPOINT_DISC)
            if epoch % 10 == 0:
                utils.save_example(g, val_loader, epoch, folder=EVAL_DIR, device=DEVICE)
        evaluate(d, g, val_loader, L1_LOSS, BCE, G_Losses, D_Losses, epoch)
    loss_log = pd.DataFrame({"G_Loss": G_Losses, "D_Loss": D_Losses})
    loss_log.to_csv('loss_log.csv')


if __name__ == "__main__":
    initialisation(True, True)
