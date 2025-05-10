import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.io import savemat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="subject ID (e.g., subj01)"
    )
    args = parser.parse_args()
    subject = args.subject

    # Define the ROIs you're combining
    roinames = ['early', 'ventral', 'midventral', 'midlateral', 'lateral', 'parietal']

    # Define VGG19 layer shapes for reshaping flat features
    maps = {
        'conv1_1': [64, 224, 224],
        'conv1_2': [64, 224, 224],
        'conv2_1': [128, 112, 112],
        'conv2_2': [128, 112, 112],
        'conv3_1': [256, 56, 56],
        'conv3_2': [256, 56, 56],
        'conv3_3': [256, 56, 56],
        'conv3_4': [256, 56, 56],
        'conv4_1': [512, 28, 28],
        'conv4_2': [512, 28, 28],
        'conv4_3': [512, 28, 28],
        'conv4_4': [512, 28, 28],
        'conv5_1': [512, 14, 14],
        'conv5_2': [512, 14, 14],
        'conv5_3': [512, 14, 14],
        'conv5_4': [512, 14, 14],
        'fc6':     [1, 4096],
        'fc7':     [1, 4096],
        'fc8':     [1, 1000],
    }

    # Set up directories
    decoded_dir = f'/shared/home/mis6559/neurobio240/decoded/{subject}/'
    output_dir = f'/shared/home/mis6559/neurobio240/decoded/gan_mod/'
    os.makedirs(output_dir, exist_ok=True)

    for layer in tqdm(maps.keys()):
        print(f'Processing Layer: {layer}')
        layer_output_dir = os.path.join(output_dir, layer, subject, 'streams')
        os.makedirs(layer_output_dir, exist_ok=True)

        # Load decoded feature scores
        feat_file = os.path.join(decoded_dir, f'{subject}_{"_".join(roinames)}_scores_{layer}.npy')
        if not os.path.exists(feat_file):
            print(f"⚠Missing: {feat_file}")
            continue

        feat = np.load(feat_file)

        for i in range(feat.shape[0]):
            reshaped_feat = feat[i].reshape(maps[layer])[np.newaxis]  # Add batch dim
            mdic = {"feat": reshaped_feat}
            save_path = os.path.join(layer_output_dir, f'VGG19-{layer}-{subject}-streams-{i:06}.mat')
            savemat(save_path, mdic)

    print("All layers saved.")


if __name__ == "__main__":
    main()
