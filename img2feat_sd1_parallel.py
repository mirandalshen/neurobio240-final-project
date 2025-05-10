import sys
sys.path.append('/shared/home/mis6559/neurobio240/codes/utils')
from nsd_access import NSDAccess

import argparse
import os
import numpy as np
from PIL import Image
from nsd_access import NSDAccess
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from einops import repeat
from tqdm import tqdm
import torch
import sys

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:", m)
    if len(u) > 0 and verbose:
        print("unexpected keys:", u)
    model.cuda(f"cuda:{gpu}")
    model.eval()
    return model

def load_img_from_arr(img_arr, resolution):
    image = Image.fromarray(img_arr).convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgidx", nargs=2, type=int, default=[0, 73000], help="Image index range: start end")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    seed_everything(args.seed)
    start_idx, end_idx = args.imgidx
    gpu = args.gpu

    resolution = 320
    batch_size = 1
    ddim_steps = 50
    ddim_eta = 0.0
    strength = 0.8
    scale = 5.0
    t_enc = int(strength * ddim_steps)

    nsd_path = '/shared/home/mis6559/neurobio240/nsd/'
    config_path = '../neurobio240/stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
    ckpt_path = '../neurobio240/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'

    nsda = NSDAccess(nsd_path)
    config = OmegaConf.load(config_path)

    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = load_model_from_config(config, ckpt_path, gpu)
    model = model.to(device)

    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    save_dir_latent = f'./nsdfeat/init_latent/'
    save_dir_cond = f'./nsdfeat/c/'
    os.makedirs(save_dir_latent, exist_ok=True)
    os.makedirs(save_dir_cond, exist_ok=True)

    precision_scope = autocast if torch.cuda.is_available() else nullcontext

    for idx in tqdm(range(start_idx, end_idx)):
        latent_path = os.path.join(save_dir_latent, f"{idx:06}.npy")
        cond_path = os.path.join(save_dir_cond, f"{idx:06}.npy")

        if os.path.exists(latent_path) and os.path.exists(cond_path):
            continue

        try:
            print(f"Processing image {idx:06}")

            prompts = nsda.read_image_coco_info([idx], info_type='captions')
            prompt = [p['caption'] for p in prompts]
            image_array = nsda.read_images(idx)
            init_image = load_img_from_arr(image_array, resolution).to(device)
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))

            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        uc = model.get_learned_conditioning(batch_size * [""])
                        c = model.get_learned_conditioning(prompt).mean(dim=0).unsqueeze(0)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale, unconditional_conditioning=uc)

            np.save(latent_path, init_latent.cpu().numpy().flatten())
            np.save(cond_path, c.cpu().numpy().flatten())

        except Exception as e:
            print(f"Error processing image {idx}: {e}")

if __name__ == "__main__":
    main()