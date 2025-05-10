import argparse
import numpy as np
import os
from tqdm import tqdm
import glob

def load_feat_org(imgid, subject, method, usefeat):
    featdir = f'/shared/home/mis6559/neurobio240/identification/cvpr_occluded/{subject}/'
    pattern = f'{featdir}/{imgid:05}_org_*_{usefeat}.npy'
    matches = glob.glob(pattern)
    if not matches:
        print(f"Skipping image {imgid:05}: no matching original feature")
        return None
    return np.load(matches[0]).flatten().squeeze()

def load_feat_gen(imgid, subject, method, usefeat, nrep=5):
    featdir = f'/shared/home/mis6559/neurobio240/identification/cvpr_occluded/{subject}/'
    feats_gen = []
    for rep in range(nrep):
        fname = f"{featdir}/{imgid:05}_org_{rep:03}_{usefeat}.npy"
        if not os.path.exists(fname):
            print(f"Missing gen feature: {fname}")
            continue
        feat_gen = np.load(fname).flatten().squeeze()
        feats_gen.append(feat_gen)
    return feats_gen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usefeat", required=True, type=str, help="e.g. inception, clip, etc.")
    parser.add_argument("--subject", required=True, type=str, help="e.g. subj01")
    parser.add_argument("--method", required=True, type=str, help="(ignored but required for consistency)")

    args = parser.parse_args()
    usefeat = args.usefeat
    subject = args.subject
    method = args.method

    nimage = 104

    print("Loading features...")
    feat_orgs = []
    feat_gens = []
    valid_indices = []

    for imgid in tqdm(range(nimage)):
        feat_org = load_feat_org(imgid, subject, method, usefeat)
        feat_gen = load_feat_gen(imgid, subject, method, usefeat)

        if feat_org is None or len(feat_gen) == 0:
            continue

        feat_orgs.append(feat_org)
        feat_gens.append(feat_gen)
        valid_indices.append(imgid)

    print(f"Loaded {len(valid_indices)} valid examples")

    print("Calculating pairwise similarity...")
    rs_all = []
    for i in tqdm(range(len(valid_indices))):
        org_feat = feat_orgs[i]
        rs_row = []
        for j in range(len(valid_indices)):
            gen_feats = feat_gens[j]
            r_vals = [np.corrcoef(org_feat, gen)[0, 1] for gen in gen_feats]
            rs_row.append(r_vals)
        rs_all.append(rs_row)

    print("Calculating identification accuracy...")
    acc_all = []
    for i in tqdm(range(len(valid_indices))):
        r_true = rs_all[i][i]
        fake_indices = [j for j in range(len(valid_indices)) if j != i]
        count = sum(np.mean(r_true) > np.mean(rs_all[i][j]) for j in fake_indices)
        acc_all.append(count / len(fake_indices))

    acc_all = np.array(acc_all)
    acc_mean = np.mean(acc_all)
    print(f"\n[OCCLUDED] {subject}_{usefeat}:   ACC = {acc_mean:.3f}")

    print("Saving per-image correlation scores...")
    per_image_scores = []
    for i in range(len(valid_indices)):
        r_true = rs_all[i][i]
        mean_score = np.mean(r_true)
        per_image_scores.append((valid_indices[i], mean_score))

    out_path = f"per_image_scores_{subject}_cvpr_occluded_{usefeat}.csv"
    with open(out_path, "w") as f:
        f.write("imgidx,mean_corr\n")
        for imgidx, score in per_image_scores:
            f.write(f"{imgidx},{score:.5f}\n")

    print(f"Per-image scores saved to: {out_path}")

if __name__ == "__main__":
    main()
