import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipModel
from torchvision import transforms

try:
    from nsd_access import NSDAccess
except ImportError:
    NSDAccess = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--imgidx", nargs=2, type=int, default=[0, 73000], help="Image index range")
    parser.add_argument("--dataset", choices=["nsd", "occluded"], default="nsd")
    parser.add_argument("--occluded_dir", type=str, default="/shared/home/mis6559/neurobio240/nsdfeat/occluded_images")
    parser.add_argument("--outdir", type=str, default="/shared/home/mis6559/neurobio240/nsdfeat/blip/")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # Load model
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval()

    # Load NSD if using full dataset
    nsda = NSDAccess('/shared/home/mis6559/neurobio240/nsd/') if args.dataset == "nsd" else None

    for idx in tqdm(range(args.imgidx[0], args.imgidx[1])):
        out_path = os.path.join(args.outdir, f"{idx:06}.npy")
        if os.path.exists(out_path):
            continue

        try:
            # Load image
            if args.dataset == "occluded":
                img_path = os.path.join(args.occluded_dir, f"{idx:06}.jpg")
                if not os.path.exists(img_path):
                    print(f"Skipping missing occluded image: {img_path}")
                    continue
                image = Image.open(img_path).convert("RGB")
            else:
                img_arr = nsda.read_images(idx)
                image = Image.fromarray(img_arr).convert("RGB")

            # Resize and prepare
            image = image.resize((224, 224), Image.LANCZOS)
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.vision_model(**inputs)
                features = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

            np.save(out_path, features)

        except Exception as e:
            print(f"Skipping {idx}: {e}")

if __name__ == "__main__":
    main()
