import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoImageProcessor, DPTForDepthEstimation
import argparse
import h5py

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("--imgidx", nargs=2, type=int, default=[0, 73000], help="Image index range: start end")
parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
parser.add_argument("--stimuli_path", type=str, required=True, help="Path to nsd_stimuli.hdf5")
args = parser.parse_args()

start_idx, end_idx = args.imgidx
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
image_size = 512
hdf5_path = args.stimuli_path

# Load DPT model
print("Loading DPT model...")
processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
model.eval()

# Output folders
outdir_depth = "/shared/home/mis6559/neurobio240/nsdfeat/dpt/"
outdirs_emb = [
    f"/shared/home/mis6559/neurobio240/nsdfeat/dpt_emb{i}/"
    for i in range(4)
]
os.makedirs(outdir_depth, exist_ok=True)
for od in outdirs_emb:
    os.makedirs(od, exist_ok=True)

# Load image from HDF5
def load_nsd_image_hdf5(hdf5_path, idx):
    with h5py.File(hdf5_path, 'r') as f:
        img = f['imgBrick'][idx]  # shape: (425, 425, 3)
        return Image.fromarray(img)

# Run extraction
for idx in tqdm(range(start_idx, end_idx)):
    out_path = os.path.join(outdir_depth, f"{idx:06}.npy")
    if os.path.exists(out_path):
        continue

    try:
        image = load_nsd_image_hdf5(hdf5_path, idx).convert("RGB").resize((image_size, image_size), resample=Image.LANCZOS)
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            depth = outputs.predicted_depth.cpu().numpy()

            hidden_states = [
                hs.cpu().numpy()
                for i, hs in enumerate(outputs.hidden_states[1:])
                if i in model.config.backbone_out_indices
            ]

        np.save(out_path, depth)
        for i, emb in enumerate(hidden_states):
            np.save(os.path.join(outdirs_emb[i], f"{idx:06}.npy"), emb)

    except Exception as e:
        print(f"Skipping {idx}: {e}")
