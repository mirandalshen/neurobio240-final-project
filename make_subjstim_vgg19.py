import numpy as np
import scipy.io
from tqdm import tqdm
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=str, required=True, help="Layer of VGG19 (e.g., conv1_1)")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., subj01)")
    args = parser.parse_args()

    layer = args.layer
    subject = args.subject

    datdir = '../../nsdfeat/vgg19_features'
    savedir = '/shared/home/mis6559/neurobio240/nsdfeat/subjfeat/'

    stimfile = f'./mrifeat/{subject}/{subject}_stims_ave.npy'
    expdesign_path = '/shared/home/mis6559/neurobio240/nsd/nsddata/experiments/nsd/nsd_expdesign.mat'

    print(f"Layer: {layer}")
    print(f"Subject: {subject}")
    print("Loading experimental design...")

    nsd_expdesign = scipy.io.loadmat(expdesign_path)
    sharedix = nsd_expdesign['sharedix'].flatten() - 1

    print("Loading stimulus list...")
    stims = np.load(stimfile)
    feats = []
    tr_idx = []
    kept_stims = []

    print("Indexing available features...")
    mat_dir = f'/shared/home/mis6559/neurobio240/nsdfeat/vgg19_features/{layer}/nsd/org/'

    existing = set([
        int(f.split('-')[-1].split('.')[0])
        for f in os.listdir(mat_dir)
        if f.endswith('.mat')
    ])
    print(f"Found {len(existing)} existing .mat files for {layer}")

    print("Loading features...")
    for stim_id in tqdm(stims):
        if stim_id not in existing:
            continue

        mat_path = f'{mat_dir}/VGG19-{layer}-nsd-org-{stim_id:06}.mat'
        feat = scipy.io.loadmat(mat_path)['feat'].flatten()
        feats.append(feat)
        tr_idx.append(0 if stim_id in sharedix else 1)
        kept_stims.append(stim_id)

    if len(feats) == 0:
        print("No features found — possibly because this layer isn't fully processed yet.")
        return

    feats = np.stack(feats)
    tr_idx = np.array(tr_idx)
    os.makedirs(savedir, exist_ok=True)

    np.save(f'{savedir}/{subject}_{layer}_stim_ids.npy', np.array(kept_stims))
    np.save(f'{savedir}/{subject}_{layer}_tr.npy', feats[tr_idx == 1])
    np.save(f'{savedir}/{subject}_{layer}_te.npy', feats[tr_idx == 0])

    print(f"Saved:\n  • {savedir}/{subject}_{layer}_stim_ids.npy\n  • {savedir}/{subject}_{layer}_tr.npy\n  • {savedir}/{subject}_{layer}_te.npy")

if __name__ == "__main__":
    main()

