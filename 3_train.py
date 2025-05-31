#%%
import os
import numpy as np
import argparse


from torch.utils.data import TensorDataset, DataLoader
# from msd_pytorch import MSDRegressionModel
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[32, 64, 128]):
        super().__init__()
        self.downs = nn.ModuleList()
        prev_ch = in_ch
        for f in features:
            self.downs.append(nn.Sequential(
                nn.Conv2d(prev_ch, f, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True),
            ))
            prev_ch = f
        self.pool = nn.MaxPool2d(2)

        # Bottleneck doubles the last feature count
        bn_ch = features[-1] * 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(prev_ch, bn_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(bn_ch,   bn_ch, 3, padding=1), nn.ReLU(inplace=True),
        )

        # Build the up path dynamically: each up block takes (curr_ch + skip_ch) in, and outputs skip_ch
        self.ups = nn.ModuleList()
        curr_ch = bn_ch
        for skip_ch in reversed(features):
            self.ups.append(nn.Sequential(
                nn.Conv2d(curr_ch + skip_ch, skip_ch, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(skip_ch, skip_ch, 3, padding=1), nn.ReLU(inplace=True),
            ))
            curr_ch = skip_ch

        # Final 1×1 conv to get back to out_ch
        self.final = nn.Conv2d(curr_ch, out_ch, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, skip in zip(self.ups, reversed(skips)):
            # match the exact spatial size of the skip (avoids odd‐size issues)
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = up(x)

        return self.final(x)

#%%
# GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

# Parse arguments
parser = argparse.ArgumentParser(description="Set SR4ZCT parameters via command line.")
parser.add_argument("-f", "--fibre", type=str2bool, default=True, help="Enable fibre mode (True/False)")
parser.add_argument("-a", "--n_axis", type=int, choices=[0, 1, 2], default=0, help="Axis index (0, 1, or 2)")
parser.add_argument("-r", "--radius", type=int, default=5, help="Radius of the cylinder")
parser.add_argument("-s", "--resolution", type=int, default=512, help="Volume resolution")
parser.add_argument("-n", "--norm", type=float, default=1.0, help="Normalization factor")

args = parser.parse_args()

# 1. Load SR4ZCT data
FIBRE = args.fibre
N_AXIS = args.n_axis
RADIUS = args.radius
RESOLUTION = args.resolution
NORM = args.norm

print("FIBRE =", FIBRE)
print("N_AXIS =", N_AXIS)
print("RADIUS =", RADIUS)
print("RESOLUTION =", RESOLUTION)
print("NORM =", NORM)

data_dir = f"sr4zct_data_{RESOLUTION}_{RADIUS}_{FIBRE}_{N_AXIS}"
saving_dir = f'result_{RESOLUTION}_{RADIUS}_{FIBRE}_{N_AXIS}_{NORM}'
Path(saving_dir).mkdir(exist_ok=True)

X_ax = np.load(f"{data_dir}/inputs_axial.npy")
Y_ax = np.load(f"{data_dir}/targets_axial.npy")
X_co = np.load(f"{data_dir}/inputs_coronal.npy")
Y_co = np.load(f"{data_dir}/targets_coronal.npy")
X_sg = np.load(f"{data_dir}/inputs_sagittal.npy")
Y_sg = np.load(f"{data_dir}/targets_sagittal.npy")

# 2. Prepare training and test tensors
X_train = np.concatenate([X_ax, X_sg], axis=0)
Y_train = np.concatenate([Y_ax, Y_sg], axis=0)
X_train = torch.from_numpy(X_train[:, None]).float() / NORM
Y_train = torch.from_numpy(Y_train[:, None]).float() / NORM

X_test  = torch.from_numpy(X_co[:, None]).float() / NORM
Y_test  = torch.from_numpy(Y_co[:, None]).float() / NORM

# 3. Create DataLoaders
train_ds = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8)

test_ds_ax = TensorDataset(X_test, Y_test)
test_loader_ax = DataLoader(test_ds_ax, batch_size=1, shuffle=False, num_workers=4)

# 4. Model, optimizer, loss
model = UNet2D(in_ch=1, out_ch=1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

vmin, vmax = float(Y_test.min()), float(Y_test.max())

#%%
for epoch in tqdm(range(1, 51)):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        # forward
        pred = model(xb)
        loss = criterion(pred, yb)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch:03d} Train Loss {epoch_loss:.4f}")

    # every 10 epochs, dump a visual
    if epoch == 1 or epoch % 1 == 0:
        model.eval()
        with torch.no_grad():
            rng = np.random.default_rng(32)
            images = rng.integers(0, len(test_loader_ax), size=5)
            n_images = len(images)

            fig, axs = plt.subplots(n_images, 3, figsize=(12, 3 * n_images), dpi=200)
            for row, each in enumerate(images):
                inp = X_test[each:each+1].to(device)
                gt  = Y_test[each].numpy()[0]
                out = model(inp).cpu().numpy()[0, 0]
                input_img = X_test[each].numpy()[0]

                # Define titles only for the top row
                titles = ["Input", "Output", "Ground Truth"]

                for col, img in enumerate([input_img, out, gt]):
                    ax = axs[row, col] if n_images > 1 else axs[col]
                    im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
                    ax.axis('off')
                    if row == 0:
                        ax.set_title(titles[col], fontsize=12)

            # Adjust layout
            plt.tight_layout()
            fig.savefig(f"{saving_dir}/inter_epoch_{epoch}.png", bbox_inches='tight')
            plt.close(fig)

            ssim_in = []
            ssim_out = []

            psnrs_in = []
            psnrs_out = []
            with torch.no_grad():
                for xb, yb in test_loader_ax:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    ssim_in.append(ssim(
                        xb.cpu().numpy().squeeze(),
                        yb.cpu().numpy().squeeze(),
                        full=True,
                        data_range=vmax - vmin
                    )[0])
                    ssim_out.append(ssim(
                        out.cpu().numpy().squeeze(),
                        yb.cpu().numpy().squeeze(),
                        full=True,
                        data_range=vmax - vmin
                    )[0])
                    psnrs_in.append(psnr(
                        xb.cpu().numpy().squeeze(),
                        yb.cpu().numpy().squeeze(),
                        data_range=vmax - vmin
                    ))
                    psnrs_out.append(psnr(
                        out.cpu().numpy().squeeze(),
                        yb.cpu().numpy().squeeze(),
                        data_range=vmax - vmin
                    ))

            print(f"Mean IN PSNR on at {epoch}: {np.mean(psnrs_in):.2f} dB")
            print(f"Mean OUT PSNR on at {epoch}: {np.mean(psnrs_out):.2f} dB")

            print(f"Mean IN SSIM on at {epoch}: {np.mean(ssim_in):.2f} dB")
            print(f"Mean OUT SSIM on at {epoch}: {np.mean(ssim_out):.2f} dB")

        # save checkpoint
        torch.save(model.state_dict(), f"{saving_dir}/unet_epoch_{epoch}.pt")

#%%
# final evaluation on axial
# model.load_state_dict(torch.load(f"{saving_dir}/unet_epoch_50.pt"))
model.eval()
ssim_in = []
ssim_out = []

psnrs_in = []
psnrs_out = []
with torch.no_grad():
    for xb, yb in test_loader_ax:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        ssim_in.append(ssim(
            xb.cpu().numpy().squeeze(),
            yb.cpu().numpy().squeeze(),
            full=True,
            data_range=vmax - vmin
        )[0])
        ssim_out.append(ssim(
            out.cpu().numpy().squeeze(),
            yb.cpu().numpy().squeeze(),
            full=True,
            data_range=vmax - vmin
        )[0])
        psnrs_in.append(psnr(
            xb.cpu().numpy().squeeze(),
            yb.cpu().numpy().squeeze(),
            data_range=vmax - vmin
        ))
        psnrs_out.append(psnr(
            out.cpu().numpy().squeeze(),
            yb.cpu().numpy().squeeze(),
            data_range=vmax - vmin
        ))

print(f"Mean IN PSNR on axial test: {np.mean(psnrs_in):.2f} dB")
print(f"Mean OUT PSNR on axial test: {np.mean(psnrs_out):.2f} dB")

print(f"Mean IN SSIM on axial test: {np.mean(ssim_in):.2f} dB")
print(f"Mean OUT SSIM on axial test: {np.mean(ssim_out):.2f} dB")
# %%
