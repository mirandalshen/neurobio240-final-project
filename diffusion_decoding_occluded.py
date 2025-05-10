import sys
sys.path.append('/shared/home/mis6559/neurobio240/stable-diffusion')

import argparse
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import trange
from torch import autocast
from omegaconf import OmegaConf
from einops import rearrange
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt, gpu):
    torch.cuda.empty_cache()
    model = instantiate_from_config(config.model)
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model = model.half().cuda(gpu)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgidx", required=True, type=int)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subject", required=True, type=str)
    parser.add_argument("--method", type=str, default="cvpr", choices=["cvpr"])
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    config_path = '/shared/home/mis6559/neurobio240/stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
    ckpt_path = '/shared/home/mis6559/neurobio240/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path, args.gpu).to(device)

    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)
    t_enc = int(0.8 * 50)

    cond_path = f"/shared/home/mis6559/neurobio240/decoded/{args.subject}/subj01_ventral_scores_c_occluded_occluded.npy"
    conds_all = np.load(cond_path)


    # Load decoded latents and conditions
    latent_file = f"/shared/home/mis6559/neurobio240/nsdfeat/init_latent/{args.imgidx:06}.npy"
    latent_arr = np.load(latent_file)
    if latent_arr.shape == (6400,):
        latent_arr = latent_arr.reshape(4, 40, 40)
    elif latent_arr.shape == (4096,):
        latent_arr = latent_arr.reshape(4, 32, 32)
    elif latent_arr.shape != (4, 32, 32) and latent_arr.shape != (4, 40, 40):
        raise ValueError(f"Unexpected latent shape: {latent_arr.shape}")
    init_latent = torch.tensor(latent_arr).unsqueeze(0).to(device).half()

    c_tensor = torch.tensor(conds_all[args.imgidx]).view(1, 77, 768).to(device).half()

    # Conditioning dictionary
    c = c_tensor
    uc = torch.zeros_like(c_tensor)


    # Output directory
    outdir = f'/shared/home/mis6559/neurobio240/decoded/image-cvpr-occluded/{args.subject}/samples/'

    os.makedirs(outdir, exist_ok=True)

    # Sampling
    base_count = 0
    for _ in trange(5, desc="Sampling"):
        with torch.no_grad(), autocast(device_type='cuda'):
            with model.ema_scope():
                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))
                samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=5.0, unconditional_conditioning=uc)
                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(outdir, f"{args.imgidx:05}_{base_count:03}.png"))
                    base_count += 1

if __name__ == "__main__":
    main()
