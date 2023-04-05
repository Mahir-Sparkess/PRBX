import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.norm_up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.norm_up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.norm_up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.norm_up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.norm_up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.norm_up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.norm_up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_norm_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.diff_up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.diff_up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.diff_up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.diff_up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.diff_up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.diff_up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.diff_up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_diff_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.rough_up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.rough_up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.rough_up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.rough_up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.rough_up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.rough_up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.rough_up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_rough_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.spec_up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.spec_up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.spec_up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.spec_up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.spec_up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.spec_up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.spec_up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_spec_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)

        norm_up1 = self.norm_up1(bottleneck)
        norm_up2 = self.norm_up2(torch.cat([norm_up1, d7], 1))
        norm_up3 = self.norm_up3(torch.cat([norm_up2, d6], 1))
        norm_up4 = self.norm_up4(torch.cat([norm_up3, d5], 1))
        norm_up5 = self.norm_up5(torch.cat([norm_up4, d4], 1))
        norm_up6 = self.norm_up6(torch.cat([norm_up5, d3], 1))
        norm_up7 = self.norm_up7(torch.cat([norm_up6, d2], 1))
        norm_final_up = self.final_norm_up(torch.cat([norm_up7, d1], 1))

        diff_up1 = self.diff_up1(bottleneck)
        diff_up2 = self.diff_up2(torch.cat([diff_up1, d7], 1))
        diff_up3 = self.diff_up3(torch.cat([diff_up2, d6], 1))
        diff_up4 = self.diff_up4(torch.cat([diff_up3, d5], 1))
        diff_up5 = self.diff_up5(torch.cat([diff_up4, d4], 1))
        diff_up6 = self.diff_up6(torch.cat([diff_up5, d3], 1))
        diff_up7 = self.diff_up7(torch.cat([diff_up6, d2], 1))
        diff_final_up = self.final_diff_up(torch.cat([diff_up7, d1], 1))

        rough_up1 = self.rough_up1(bottleneck)
        rough_up2 = self.rough_up2(torch.cat([rough_up1, d7], 1))
        rough_up3 = self.rough_up3(torch.cat([rough_up2, d6], 1))
        rough_up4 = self.rough_up4(torch.cat([rough_up3, d5], 1))
        rough_up5 = self.rough_up5(torch.cat([rough_up4, d4], 1))
        rough_up6 = self.rough_up6(torch.cat([rough_up5, d3], 1))
        rough_up7 = self.rough_up7(torch.cat([rough_up6, d2], 1))
        rough_final_up = self.final_rough_up(torch.cat([rough_up7, d1], 1))

        spec_up1 = self.spec_up1(bottleneck)
        spec_up2 = self.spec_up2(torch.cat([spec_up1, d7], 1))
        spec_up3 = self.spec_up3(torch.cat([spec_up2, d6], 1))
        spec_up4 = self.spec_up4(torch.cat([spec_up3, d5], 1))
        spec_up5 = self.spec_up5(torch.cat([spec_up4, d4], 1))
        spec_up6 = self.spec_up6(torch.cat([spec_up5, d3], 1))
        spec_up7 = self.spec_up7(torch.cat([spec_up6, d2], 1))
        spec_final_up = self.final_spec_up(torch.cat([spec_up7, d1], 1))

        return torch.cat([norm_final_up, diff_final_up, rough_final_up, spec_final_up], 2)


def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
