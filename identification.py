import argparse
import numpy as np
from tqdm import tqdm
import os

def load_feat_org(imgid, subject, method, usefeat):
    path = f'/shared/home/mis6559/neurobio240/identification/{method}/{subject}/{imgid:05}_org_{usefeat}.npy'
    return np.load(path).flatten()

def load_feat_gen(imgid, subject, method, usefeat):
    path_template = f'/shared/home/mis6559/neurobio240/identification/{method}/{subject}/{imgid:05}_{{rep:03}}_{usefeat}.npy'
    feats = [np.load(path_template.format(rep=rep)).flatten() for rep in range(5)]
    return feats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usefeat", required=True, help="e.g., inception, alexnet5, clip")
    parser.add_argument("--subject", required=True, help="e.g., subj01")
    parser.add_argument("--method", required=True, help="e.g., cvpr or depth")
    args = parser.parse_args()

    subject = args.subject
    method = args.method
    usefeat = args.usefeat

    # Automatically determine max nimage based on decoded features
    decoded_dir = f"/shared/home/mis6559/neurobio240/identification/{method}/{subject}/"
    all_decoded = sorted([
        int(f.split("_")[0])
        for f in os.listdir(decoded_dir)
        if f.endswith(f"_000_{usefeat}.npy")
    ])
    unique_imgids = sorted(set(all_decoded))
    print(f"Found {len(unique_imgids)} image(s) with decoded features for method: {method}")
    
    valid_indices = []
    feat_orgs = []
    feat_gens = []

    print("Loading features...")
    for i in tqdm(unique_imgids):
        try:
            org = load_feat_org(i, subject, method, usefeat)
            gen = load_feat_gen(i, subject, method, usefeat)
            feat_orgs.append(org)
            feat_gens.append(gen)
            valid_indices.append(i)
        except FileNotFoundError:
            continue

    nimage = len(valid_indices)
    print(f"Using {nimage} valid samples with available decoded + original features")

    print("Computing similarity matrix...")
    rs_all = []
    for i in tqdm(range(nimage)):
        org_feat = feat_orgs[i]
        rs_row = []
        for j in range(nimage):
            gen_feats = feat_gens[j]
            r_vals = [np.corrcoef(org_feat, gen)[0, 1] for gen in gen_feats]
            rs_row.append(r_vals)
        rs_all.append(rs_row)

    print("Computing identification accuracy...")
    acc_all = []
    for i in tqdm(range(nimage)):
        r_true = rs_all[i][i]
        fake_indices = [j for j in range(nimage) if j != i]
        count = sum(np.mean(r_true) > np.mean(rs_all[i][j]) for j in fake_indices)
        acc_all.append(count / (nimage - 1))

    acc_all = np.array(acc_all)
    acc_mean = np.mean(acc_all)
    print(f"\n{subject} | {method} | {usefeat} — Identification Accuracy: {acc_mean:.3f}")

    # Save per-image scores=
    per_image_scores = []
    for i in range(nimage):
        r_true = rs_all[i][i]
        mean_score = np.mean(r_true)
        per_image_scores.append((valid_indices[i], mean_score))
    
    # Save to CSV
    out_path = f"per_image_scores_{subject}_{method}_{usefeat}.csv"
    with open(out_path, "w") as f:
        f.write("imgidx,mean_corr\n")
        for imgidx, score in per_image_scores:
            f.write(f"{imgidx},{score:.5f}\n")
    
    print(f"Per-image correlation scores saved to {out_path}")


if __name__ == "__main__":
    main()
