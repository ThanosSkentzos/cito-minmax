import os
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from pcfv import train_loop, set_normalization, plot_images
# from msd_pytorch import MSDRegressionModel
from skimage.metrics import peak_signal_noise_ratio as psnr
from tifffile import imwrite
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

saving_dir = 'L291_result'
Path(saving_dir).mkdir(exist_ok=True)

# 1. Load SR4ZCT data
data_dir = 'sr4zct_data'
X_ax = np.load(f"{data_dir}/inputs_axial.npy")
Y_ax = np.load(f"{data_dir}/targets_axial.npy")
X_co = np.load(f"{data_dir}/inputs_coronal.npy")
Y_co = np.load(f"{data_dir}/targets_coronal.npy")
X_sg = np.load(f"{data_dir}/inputs_sagittal.npy")
Y_sg = np.load(f"{data_dir}/targets_sagittal.npy")

# 2. Prepare training and test tensors
X_train = np.concatenate([X_co, X_sg], axis=0)
Y_train = np.concatenate([Y_co, Y_sg], axis=0)
X_train = torch.from_numpy(X_train[:, None]).float()
Y_train = torch.from_numpy(Y_train[:, None]).float()

X_test  = torch.from_numpy(X_ax[:, None]).float()
Y_test  = torch.from_numpy(Y_ax[:, None]).float()

# 3. Create DataLoaders
train_ds = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8)

test_ds_ax = TensorDataset(X_test, Y_test)
test_loader_ax = DataLoader(test_ds_ax, batch_size=1, shuffle=False, num_workers=4)

# 4. Model, optimizer, loss
model = UNet2D(in_ch=1, out_ch=1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

inter_x = X_train[10].numpy()[0]
inter_y = Y_train[10].numpy()[0]
inter_x_cuda = torch.from_numpy(inter_x[None, None]).to(device)
vmin, vmax = inter_y.min(), inter_y.max()

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
    if epoch == 1 or epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            inp = X_train[10:11].to(device)   # pick one sample
            gt  = Y_train[10].numpy()[0]
            out = model(inp).cpu().numpy()[0,0]
        fig = plot_images(
            X_train[10].numpy()[0], out, gt,
            style=plt.gray(), t1="in", t2="out", t3="gt",
            vmin=gt.min(), vmax=gt.max(), show_image=False
        )
        fig.savefig(f"{saving_dir}/inter_epoch_{epoch}.png")

    # save checkpoint
    torch.save(model.state_dict(), f"{saving_dir}/unet_epoch_{epoch}.pt")

# final evaluation on axial
model.eval()
psnrs = []
with torch.no_grad():
    for xb, yb in test_loader_ax:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        psnrs.append(psnr(
            out.cpu().numpy().squeeze(),
            yb.cpu().numpy().squeeze(),
            data_range=vmax - vmin
        ))
# save one example
imwrite(
    f"{saving_dir}/axial_pred.tif",
    model(X_test[0:1].to(device)).detach().cpu().numpy()[0,0]
)

print(f"Mean PSNR on axial test: {np.mean(psnrs):.2f} dB")


# # 6. Training Loop
# for epoch in range(1, 201):
#     loss = train_loop(train_loader, model, optimizer, criterion, device)
#     print(f"Epoch {epoch:03d} Loss {loss:.4f}")
#     if epoch == 1 or epoch % 10 == 0:
#         pred = model(inter_x_cuda).cpu().detach().numpy()[0,0]
#         fig = plot_images(inter_x, pred, inter_y,
#                             style=plt.gray(), t1="input", t2="pred", t3="gt",
#                             vmin=vmin, vmax=vmax, show_image=False)
#         fig.savefig(f"{saving_dir}/inter_epoch_{epoch}.png")
#     torch.save(model.state_dict(), f"{saving_dir}/sr_epoch_{epoch}.pt")

# # 7. Evaluate on axial (unseen) slices
# psnrs = []
# for xb, yb in test_loader_ax:
#     xb, yb = xb.to(device), yb.to(device)
#     with torch.no_grad():
#         out = model(xb)
#     psnrs.append(psnr(out.cpu().numpy().squeeze(),
#                         yb.cpu().numpy().squeeze(),
#                         data_range=vmax-vmin))

# # Save one example prediction
# pred_ax = model(X_test[0:1].to(device)).cpu().numpy()[0,0]
# imwrite(f"{saving_dir}/axial_pred.tif", pred_ax)

# print(f"Mean PSNR on axial test: {np.mean(psnrs):.2f} dB")
