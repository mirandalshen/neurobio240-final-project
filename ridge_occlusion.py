import argparse
import os
import numpy as np
import scipy.io
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.scoring import correlation_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--roi', type=str, nargs='+', required=True)
    parser.add_argument('--use_stim', type=str, default='each', choices=['each', 'ave'])
    parser.add_argument('--dataset', choices=['original', 'occluded'], default='original')
    args = parser.parse_args()

    subject = args.subject
    target = args.target
    roi_list = args.roi
    use_stim = args.use_stim
    dataset = args.dataset

    # Setup paths
    mridir = f"/shared/home/mis6559/neurobio240/mrifeat/{subject}/"
    savedir = f"/shared/home/mis6559/neurobio240/decoded/{subject}/"
    os.makedirs(savedir, exist_ok=True)

    # All feature files are saved in the same directory
    featdir = "/shared/home/mis6559/neurobio240/nsdfeat/subjfeat/"

    # Setup model
    backend = set_backend("numpy", on_error="warn")
    if target in ["init_latent", "c"]:
        alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
    else:
        alphas = [10000, 20000, 40000]

    ridge = RidgeCV(alphas=alphas)

    pipeline = make_pipeline(StandardScaler(), ridge)

    # Load features
    if dataset == "occluded":
        base = f"{subject}_{use_stim}_{target}"
    elif target.startswith("conv") or target.startswith("fc"):
        base = f"{subject}_{use_stim}_{target}"
    else:
        base = f"{subject}_{use_stim}_{target}"
    
    stim_ids = np.load(f"{featdir}/{base}_stim_ids.npy")
    Y = np.load(f"{featdir}/{base}_tr.npy").astype("float32")
    if Y.ndim > 2:
        print(f"Reshaping Y from {Y.shape} to 2D")
        Y = Y.reshape(Y.shape[0], -1)

    stim_to_trial = np.load(f"{mridir}/{subject}_stims.npy")
    stim_id_to_trial_idx = {stim: i for i, stim in enumerate(stim_to_trial)}
    trial_indices = [stim_id_to_trial_idx[s] for s in stim_ids if s in stim_id_to_trial_idx]

    X_list = []
    for roi in roi_list:
        full_X = np.load(f"{mridir}/{subject}_{roi}_betas_tr.npy").astype("float32")
        filtered_indices = [i for i in trial_indices if i < full_X.shape[0]]
        X_list.append(full_X[filtered_indices])
    X = np.hstack(X_list)

    # Align
    min_len = min(X.shape[0], Y.shape[0])
    X = X[:min_len]
    Y = Y[:min_len]

    # Test data
    stim_to_row_te = np.load(f"{mridir}/{subject}_stims_ave.npy")
    expdesign = scipy.io.loadmat('/shared/home/mis6559/neurobio240/nsd/nsddata/experiments/nsd/nsd_expdesign.mat')
    sharedix = expdesign['sharedix'].flatten() - 1
    stim_ids_te = [s for s in stim_ids if s in sharedix and s in set(stim_to_row_te)]
    stim_id_to_test_row = {stim: i for i, stim in enumerate(stim_to_row_te)}
    row_indices_te = [stim_id_to_test_row[s] for s in stim_ids_te if s in stim_id_to_test_row and stim_id_to_test_row[s] < 982]

    X_te = np.hstack([
        np.load(f"{mridir}/{subject}_{roi}_betas_ave_te.npy")[row_indices_te]
        for roi in roi_list
    ])

    # Load test features
    if dataset == "original":
        test_feat_file = f"{featdir}/{base}_te.npy"
    else:
        test_feat_file = f"{featdir}/{base}_te.npy"

    Y_te = np.load(test_feat_file).astype("float32")
    if Y_te.ndim > 2:
        print(f"Reshaping Y_te from {Y_te.shape} to 2D")
        Y_te = Y_te.reshape(Y_te.shape[0], -1)
    Y_te = Y_te[:X_te.shape[0]]

    # Fit model
    print(f"Training model for {subject} | ROIs: {roi_list} | Target: {target}")
    print(f"Train shapes: X={X.shape}, Y={Y.shape}")
    print(f"Test  shapes: X_te={X_te.shape}, Y_te={Y_te.shape}")

    pipeline.fit(X, Y)
    Y_pred = pipeline.predict(X_te)

    r = correlation_score(Y_te.T, Y_pred.T)
    r = r.cpu().numpy() if hasattr(r, "cpu") else r
    print(f"Mean test correlation (r): {np.mean(r):.4f}")

    roi_str = "_".join(roi_list)
    tag = f"{target}_{dataset}"
    out_file = f"{savedir}/{subject}_{roi_str}_scores_{tag}.npy"
    np.save(out_file, Y_pred)
    print(f"Saved predictions to: {out_file}")

if __name__ == "__main__":
    main()