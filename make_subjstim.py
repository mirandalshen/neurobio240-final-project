import argparse
import numpy as np
import scipy.io
import os
from tqdm import tqdm

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, required=True, help="e.g., subj01")
parser.add_argument("--featname", type=str, required=True, help="e.g., init_latent, c, blip, dpt_emb0")
parser.add_argument("--use_stim", type=str, choices=["each", "ave"], default="ave", help="Use averaged (ave) or all (each) stimuli")
args = parser.parse_args()

subject = args.subject
featname = args.featname
use_stim = args.use_stim

# Paths
topdir = './nsdfeat/'
featdir = os.path.join(topdir, featname)
savedir = os.path.join(topdir, 'subjfeat')
os.makedirs(savedir, exist_ok=True)

# Load shared index
expdesign = scipy.io.loadmat('/shared/home/mis6559/neurobio240/nsd/nsddata/experiments/nsd/nsd_expdesign.mat')
sharedix = expdesign['sharedix'].flatten() - 1  # 0-based indexing

# Load stimulus list for subject
stim_path = f'./mrifeat/{subject}/{subject}_stims_ave.npy' if use_stim == "ave" else f'./mrifeat/{subject}/{subject}_stims.npy'
stims = np.load(stim_path)
stims = stims[stims >= 0]

# Filter to existing feature files
existing_feat_ids = set(
    int(fname.split('.')[0])
    for fname in os.listdir(featdir)
    if fname.endswith('.npy')
)
stims = np.array([s for s in stims if s in existing_feat_ids])
print(f"Kept {len(stims)} stimuli with existing {featname} features.")

# Determine feature shape from first available feature
first_feat = np.load(f"{featdir}/{stims[0]:06}.npy")
feat_shape = first_feat.shape
print("Feature shape per stimulus:", feat_shape)

# Allocate memory-mapped array
feats_path = os.path.join(savedir, f"{subject}_{use_stim}_{featname}_all.memmap")
feats = np.memmap(feats_path, dtype=np.float32, mode='w+', shape=(len(stims), *feat_shape))

# Create training/test split
tr_idx = np.zeros(len(stims), dtype=int)

# Load features into memmap
for idx, stim_id in tqdm(enumerate(stims), total=len(stims)):
    try:
        feat = np.load(f"{featdir}/{stim_id:06}.npy")
        feats[idx] = feat
        tr_idx[idx] = 0 if stim_id in sharedix else 1
    except Exception as e:
        print(f"Error with stimulus {stim_id:06}: {e}")
        feats[idx] = 0
        tr_idx[idx] = -1

feats.flush()

# Reload in read-only mode to split
feats = np.memmap(feats_path, dtype=np.float32, mode='r', shape=(len(stims), *feat_shape))
feats_tr = feats[tr_idx == 1]
feats_te = feats[tr_idx == 0]

# Save outputs
np.save(f"./mrifeat/{subject}/{subject}_stims_tridx.npy", tr_idx)
np.save(f"{savedir}/{subject}_{use_stim}_{featname}_tr.npy", feats_tr)
np.save(f"{savedir}/{subject}_{use_stim}_{featname}_te.npy", feats_te)
np.save(f"{savedir}/{subject}_{use_stim}_{featname}_stim_ids.npy", stims)

# Done
print("Finished! Saved:")
print(f"  → {feats_tr.shape} → {subject}_{use_stim}_{featname}_tr.npy")
print(f"  → {feats_te.shape} → {subject}_{use_stim}_{featname}_te.npy")
