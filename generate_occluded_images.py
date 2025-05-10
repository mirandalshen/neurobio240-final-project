import os
import h5py
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import argparse

def apply_occlusion(image, mask_type="rectangle", occlusion_ratio=0.3):
    """Applies a simple occlusion (e.g., black rectangle) to the image."""
    w, h = image.size
    draw = ImageDraw.Draw(image)

    if mask_type == "rectangle":
        ow, oh = int(w * occlusion_ratio), int(h * occlusion_ratio)
        left = (w - ow) // 2
        top = (h - oh) // 2
        draw.rectangle([left, top, left + ow, top + oh], fill=(0, 0, 0))

    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, required=True, help='Path to nsd_stimuli.hdf5')
    parser.add_argument('--outdir', type=str, default='./occluded_images', help='Directory to save occluded images')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1000)
    parser.add_argument('--image_size', type=int, default=425)
    parser.add_argument('--occlusion_ratio', type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading HDF5 from: {args.h5_path}")
    with h5py.File(args.h5_path, 'r') as f:
        dataset = f['imgBrick']
        total = dataset.shape[0]
        print(f"Total images in dataset: {total}")
        end = min(args.end, total)

        for idx in tqdm(range(args.start, end)):
            img = dataset[idx].astype(np.uint8)
            image = Image.fromarray(img).convert("RGB")
            image = image.resize((args.image_size, args.image_size), Image.LANCZOS)

            occluded = apply_occlusion(image, occlusion_ratio=args.occlusion_ratio)
            save_path = os.path.join(args.outdir, f"{idx:06}.jpg")
            occluded.save(save_path)

    print("Finished generating occluded images.")

if __name__ == "__main__":
    main()
