import os
import numpy as np
import argparse
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, type=str)
    parser.add_argument("--target", required=True, type=str)
    parser.add_argument("--rois", nargs='+', required=True, help="List of ROIs to combine")
    parser.add_argument("--decoded_dir", type=str, default="/shared/home/mis6559/neurobio240/decoded")
    args = parser.parse_args()

    subject = args.subject
    target = args.target
    roi_list = args.rois
    decoded_dir = args.decoded_dir

    print(f"Combining predictions for subject: {subject}")
    print(f"ROIs: {roi_list}")

    y_pred_list = []
    found_rois = []
    shapes_encountered = set()

    all_files = glob.glob(os.path.join(decoded_dir, subject, f"{subject}_*_scores_{target}.npy"))

    for roi in roi_list:
        matched = False
        for path in all_files:
            filename = os.path.basename(path)
            if f"_{roi}_" in filename or filename.startswith(f"{subject}_{roi}_"):
                y_pred = np.load(path)
                print(f"Loaded: {path} with shape {y_pred.shape}")

                if not y_pred_list:
                    expected_shape = y_pred.shape
                elif y_pred.shape != expected_shape:
                    print(f"⚠Skipped {roi}: shape {y_pred.shape} does not match expected {expected_shape}")
                    continue

                y_pred_list.append(y_pred)
                found_rois.append(roi)
                matched = True
                break
        if not matched:
            print(f"Missing ROI: {roi}")

    if not y_pred_list:
        raise ValueError("No valid ROI predictions found. Check paths and filenames.")

    Y_combined = np.mean(np.stack(y_pred_list), axis=0)
    roi_str = "_".join(found_rois)

    save_path = os.path.join(decoded_dir, subject, f"{subject}_{roi_str}_scores_{target}_combined.npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, Y_combined)

    print(f"Combined predictions saved to: {save_path}")
    print(f"Final shape: {Y_combined.shape}")

if __name__ == "__main__":
    main()
