import sys
sys.path.append('/shared/home/mis6559/neurobio240/stable-diffusion')

import argparse
import os
import h5py
import numpy as np
import pandas as pd
import scipy.io
from PIL import Image
from tqdm import trange
import torch
from torch import autocast
from omegaconf import OmegaConf
from einops import rearrange
import s3fs
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

def load_img_from_arr(img_arr):
    image = Image.fromarray(img_arr).convert("RGB").resize((512, 512), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return 2. * torch.from_numpy(image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgidx", required=True, type=int)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subject", required=True, type=str)
    parser.add_argument("--method", required=True, type=str)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load NSD stimuli
    local_path = '/shared/home/mis6559/neurobio240/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5'
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        fs = s3fs.S3FileSystem()
        fs.get('natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', local_path)

    sf = h5py.File(local_path, 'r')
    sdataset = sf.get('imgBrick')
    stims_ave = np.load(f'/shared/home/mis6559/neurobio240/mrifeat/{args.subject}/{args.subject}_stims_ave.npy')

    nsd_expdesign = scipy.io.loadmat('/shared/home/mis6559/neurobio240/nsd/nsddata/experiments/nsd/nsd_expdesign.mat')
    sharedix = nsd_expdesign['sharedix'].flatten() - 1
    tr_idx = np.isin(stims_ave, sharedix).astype(int) ^ 1
    imgidx_te = np.where(tr_idx == 0)[0][args.imgidx]
    idx73k = stims_ave[imgidx_te]

    # Save original image
    outdir = f'/shared/home/mis6559/neurobio240/decoded/image-{args.method}/{args.subject}/samples/'
    os.makedirs(outdir, exist_ok=True)
    Image.fromarray(np.squeeze(sdataset[idx73k]).astype(np.uint8)).save(
        os.path.join(outdir, f"{args.imgidx:05}_org.png"))

    # Select model config + checkpoint
    if args.method == "depth":
        config_path = '/shared/home/mis6559/neurobio240/stable-diffusion/configs/stable-diffusion/v2-midas-inference.yaml'
        ckpt_path = '/shared/home/mis6559/neurobio240/stable-diffusion/models/ldm/stable-diffusion-v2/sd-v2-depth.ckpt'
    elif args.method == "text":
        config_path = '/shared/home/mis6559/neurobio240/stable-diffusion/configs/stable-diffusion/v2-inference.yaml'
        ckpt_path = '/shared/home/mis6559/neurobio240/stable-diffusion/models/ldm/stable-diffusion-v2/sd-v2-512.ckpt'
    elif args.method == "cvpr":
        config_path = '/shared/home/mis6559/neurobio240/stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
        ckpt_path = '/shared/home/mis6559/neurobio240/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'
    else:
        raise NotImplementedError(f"Unsupported method: {args.method}")

    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path, args.gpu).to(device)

    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)
    t_enc = int(0.8 * 50)

    # Initialize latent
    im = Image.new("RGB", (512, 512), color="gray")
    init_image = load_img_from_arr(np.array(im)).to(device).half()
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))

    # Conditioning
    if args.method == "depth" or args.method == "text":
        cap_path = f'/shared/home/mis6559/neurobio240/decoded/{args.subject}/captions/captions_brain.csv'
        captions = pd.read_csv(cap_path, sep='\t', header=None)
        prompt = captions.iloc[args.imgidx][0]
        print(f"Prompt: \"{prompt}\"")
        c_text = model.get_learned_conditioning(prompt).to(device)

        if args.method == "depth":
            depth_path = f'/shared/home/mis6559/neurobio240/decoded/{args.subject}/dpt_fromemb/{args.imgidx:06}.npy'
            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"Missing depth file: {depth_path}")
            depth_tensor = torch.tensor(np.load(depth_path)).float().to(device)
            print("Loaded decoded depth latent")
            c = {"c_crossattn": [c_text], "c_concat": [depth_tensor]}
            uc = {"c_crossattn": [model.get_learned_conditioning([""])], "c_concat": [torch.zeros_like(depth_tensor)]}
        else:
            c = {"c_crossattn": [c_text]}
            uc = {"c_crossattn": [model.get_learned_conditioning([""])]}

    elif args.method == "cvpr":
        latent_path = f'/shared/home/mis6559/neurobio240/decoded/{args.subject}/init_latent_te.npy'
        c_path = f'/shared/home/mis6559/neurobio240/decoded/{args.subject}/c_te.npy'
        stim_ids = np.load(f'/shared/home/mis6559/neurobio240/decoded/{args.subject}/init_latent_stim_ids.npy')
        row = np.where(stim_ids == idx73k)[0][0]
        init_latent = torch.tensor(np.load(latent_path)[row]).reshape(1, 4, 32, 32).to(device).half()
        c_tensor = np.load(c_path)[row]
        c = model.get_learned_conditioning("placeholder")  # dummy to get shape
        c = c[0].unsqueeze(0).repeat(1, 1, 1)  # ensure shape
        c[0] = torch.tensor(c_tensor).to(device).half()
        uc = model.get_learned_conditioning([""])

    # Decode image
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
