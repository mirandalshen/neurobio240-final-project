import argparse, os
from tqdm import tqdm
import torch
import numpy as np
from transformers import DPTForDepthEstimation
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, type=int, help="gpu")
    parser.add_argument("--subject", required=True, type=str, help="subject name (e.g. subj01)")
    opt = parser.parse_args()

    gpu = opt.gpu
    subject = opt.subject
    roi = ['early', 'ventral', 'midventral', 'midlateral', 'lateral', 'parietal']

    datdir = f'/shared/home/mis6559/neurobio240/decoded/{subject}/'

    savedir = f'{datdir}/dpt_fromemb/'
    os.makedirs(savedir, exist_ok=True)
    savedir_img = f'{datdir}/dpt_fromemb_image/'
    os.makedirs(savedir_img, exist_ok=True)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
    model.eval()

    imsize = (512, 512)
    latentsize = (64, 64)

    # Load all 4 embedding arrays
    dpt_embs = []
    for idx in range(4):
        fname = f'{datdir}/{subject}_{"_".join(roi)}_scores_dpt_emb{idx}.npy'
        dpt_embs.append(np.load(fname))
    dpt_embs = np.stack(dpt_embs)  # Shape: (4, N, 577*1024)
    dpt_embs = torch.Tensor(dpt_embs).to(device)

    for s in tqdm(range(dpt_embs.shape[1])):
        hidden_states = [dpt_embs[idx, s, :].reshape(1, 577, 1024) for idx in range(4)]

        with torch.no_grad():
            hidden_states = model.neck(hidden_states)
            predicted_depth = model.head(hidden_states)

        # Resize for visual output
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=imsize,
            mode="bicubic",
            align_corners=False
        )

        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)

        # Resize for latent SD2 input
        cc = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=latentsize,
            mode="bicubic",
            align_corners=False
        )
        depth_min = torch.amin(cc, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(cc, dim=[1, 2, 3], keepdim=True)
        cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.

        # Save outputs
        np.save(f'{savedir}/{s:06}.npy', cc.cpu().numpy())
        depth.save(f'{savedir_img}/{s:06}.png')

if __name__ == "__main__":
    main()
