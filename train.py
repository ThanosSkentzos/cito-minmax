import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pcfv import train_loop, set_normalization, plot_images
from msd_pytorch import MSDRegressionModel
from skimage.metrics import peak_signal_noise_ratio as psnr
from tifffile import imwrite
import matplotlib.pyplot as plt
from pathlib import Path

# GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

saving_dir = 'L291_result'
Path(saving_dir).mkdir(exist_ok=True)

# 1. Load SR4ZCT data
data_dir = 'sr4zct_data'
X_ax = np.load(f"{data_dir}/inputs_axial.npy")      # (200,100,100)
Y_ax = np.load(f"{data_dir}/targets_axial.npy")
X_co = np.load(f"{data_dir}/inputs_coronal.npy")    # (200,100,100)
Y_co = np.load(f"{data_dir}/targets_coronal.npy")
X_sg = np.load(f"{data_dir}/inputs_sagittal.npy")   # (200,100,100)
Y_sg = np.load(f"{data_dir}/targets_sagittal.npy")

# 2. Prepare training and test tensors
X_train = np.concatenate([X_co, X_sg], axis=0)      # (400,100,100)
Y_train = np.concatenate([Y_co, Y_sg], axis=0)
X_train = torch.from_numpy(X_train[:, None]).float()
Y_train = torch.from_numpy(Y_train[:, None]).float()

X_test  = torch.from_numpy(X_ax[:, None]).float()   # (200,1,100,100)
Y_test  = torch.from_numpy(Y_ax[:, None]).float()

# 3. Create DataLoaders
train_ds = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8)

test_ds_ax = TensorDataset(X_test, Y_test)
test_loader_ax = DataLoader(test_ds_ax, batch_size=1, shuffle=False, num_workers=4)

# 4. Model, optimizer, loss
model = MSDRegressionModel(1, 1, 100, 1, dilations=list(range(1,11)))
set_normalization(model, train_loader)
model = model.net.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 5. (Optional) Pick one sample to visualize
inter_x = X_train[10].numpy()[0]
inter_y = Y_train[10].numpy()[0]
inter_x_cuda = torch.from_numpy(inter_x[None, None]).to(device)
vmin, vmax = inter_y.min(), inter_y.max()

# 6. Training Loop
for epoch in range(1, 201):
    loss = train_loop(train_loader, model, optimizer, criterion, device)
    print(f"Epoch {epoch:03d} Loss {loss:.4f}")
    if epoch == 1 or epoch % 10 == 0:
        pred = model(inter_x_cuda).cpu().detach().numpy()[0,0]
        fig = plot_images(inter_x, pred, inter_y,
                            style=plt.gray(), t1="input", t2="pred", t3="gt",
                            vmin=vmin, vmax=vmax, show_image=False)
        fig.savefig(f"{saving_dir}/inter_epoch_{epoch}.png")
    torch.save(model.state_dict(), f"{saving_dir}/sr_epoch_{epoch}.pt")

# 7. Evaluate on axial (unseen) slices
psnrs = []
for xb, yb in test_loader_ax:
    xb, yb = xb.to(device), yb.to(device)
    with torch.no_grad():
        out = model(xb)
    psnrs.append(psnr(out.cpu().numpy().squeeze(),
                        yb.cpu().numpy().squeeze(),
                        data_range=vmax-vmin))

# Save one example prediction
pred_ax = model(X_test[0:1].to(device)).cpu().numpy()[0,0]
imwrite(f"{saving_dir}/axial_pred.tif", pred_ax)

print(f"Mean PSNR on axial test: {np.mean(psnrs):.2f} dB")
