#%%
import pandas as pd
import re
import glob

files = glob.glob("nohup_train*.out")

df_all = pd.DataFrame(columns=["Name", "Epoch", "Train_Loss", "IN_PSNR", "OUT_PSNR", "IN_SSIM", "OUT_SSIM", "PSNR_Diff"])
df_best = pd.DataFrame(columns=["Name", "Epoch", "Train_Loss", "IN_PSNR", "OUT_PSNR", "IN_SSIM", "OUT_SSIM", "PSNR_Diff"])
df_last = pd.DataFrame(columns=["Name", "Epoch", "Train_Loss", "IN_PSNR", "OUT_PSNR", "IN_SSIM", "OUT_SSIM", "PSNR_Diff"])

#%%
for each in files:
    # Compile regex patterns
    epoch_pattern = re.compile(r"Epoch\s+(\d+)\s+Train Loss ([\d.]+)")
    psnr_in_pattern = re.compile(r"Mean IN PSNR on at \d+: ([\d.]+) dB")
    psnr_out_pattern = re.compile(r"Mean OUT PSNR on at \d+: ([\d.]+) dB")
    ssim_in_pattern = re.compile(r"Mean IN SSIM on at \d+: ([\d.]+) dB")
    ssim_out_pattern = re.compile(r"Mean OUT SSIM on at \d+: ([\d.]+) dB")

    # Store results in a list of dictionaries
    records = []

    with open(each, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if (epoch_match := epoch_pattern.search(line)):
            epoch = int(epoch_match.group(1))
            train_loss = float(epoch_match.group(2))

            psnr_in = float(psnr_in_pattern.search(lines[i+1]).group(1))
            psnr_out = float(psnr_out_pattern.search(lines[i+2]).group(1))
            ssim_in = float(ssim_in_pattern.search(lines[i+3]).group(1))
            ssim_out = float(ssim_out_pattern.search(lines[i+4]).group(1))
            
            records.append({
                "Name": each.replace("nohup_train_", "").replace(".out", ""),
                "Epoch": epoch,
                "Train_Loss": train_loss,
                "IN_PSNR": psnr_in,
                "OUT_PSNR": psnr_out,
                "IN_SSIM": ssim_in,
                "OUT_SSIM": ssim_out,
            })
            i += 5
        else:
            i += 1

    # Create DataFrame
    df = pd.DataFrame(records)
    df["PSNR_Diff"] = df["OUT_PSNR"] - df["IN_PSNR"]
    
    best_epoch = df.loc[[df['OUT_PSNR'].idxmax()]]
    last_epoch = df.iloc[[-1]]  
    df_best = pd.concat([df_best, best_epoch], ignore_index=True)
    df_last = pd.concat([df_last, last_epoch], ignore_index=True)
    df_all = pd.concat([df_all, df], ignore_index=True)

#%%
def parse_name(name):
    parts = name.split('_')
    resolution = int(parts[0])
    radius = int(parts[1])
    fibre = parts[2] == 't'
    axis = { "0": "X", "1": "Y", "2": "Z" }[parts[3]]
    norm = "Yes" if parts[4] == "255" else "No"
    return {
        # "Resolution": int(parts[0]),
        # "Radius": int(parts[1]),
        # "Fibre": parts[2] == 't',
        # "Axis": { "0": "X", "1": "Y", "2": "Z" }[parts[3]],
        # "Norm": int(parts[4]),
        "Readable": f"Radius {radius} | Fibre: {fibre} | Axis: {axis} | Norm: {norm}"
    }

parsed = df_best["Name"].apply(parse_name)
parsed_df = pd.DataFrame(parsed.tolist())

df_best = pd.concat([df_best, parsed_df], axis=1)
cols = df_best.columns.tolist()
cols.insert(0, cols.pop(cols.index('Readable')))
df_best = df_best[cols]
df_best = df_best.drop(columns=['Name'])

parsed = df_last["Name"].apply(parse_name)
parsed_df = pd.DataFrame(parsed.tolist())

df_last = pd.concat([df_last, parsed_df], axis=1)
cols = df_last.columns.tolist()
cols.insert(0, cols.pop(cols.index('Readable')))
df_last = df_last[cols]
df_last = df_last.drop(columns=['Name'])

parsed = df_all["Name"].apply(parse_name)
parsed_df = pd.DataFrame(parsed.tolist())
# Combine with original df if needed
df_all = pd.concat([df_all, parsed_df], axis=1)
cols = df_all.columns.tolist()
cols.insert(0, cols.pop(cols.index('Readable')))
df_all = df_all[cols]
df_all = df_all.drop(columns=['Name'])
df_all.to_csv("all_epochs.csv", index=False)

# %%
df_sorted = df_best.sort_values(by="PSNR_Diff", ascending=False)
df_sorted.to_csv("best_epochs.csv", index=False)
df_sorted

#%%
print(df_sorted)

# %%
df_sorted = df_last.sort_values(by="PSNR_Diff", ascending=False)
df_sorted.to_csv("last_epochs.csv", index=False)
df_sorted

# %%
print(df_sorted)

# %%
