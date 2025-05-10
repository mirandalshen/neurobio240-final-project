import sys
sys.path.append('/shared/home/mis6559/neurobio240/BLIP')

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import logging
from BLIP.models.blip import blip_decoder

logging.set_verbosity_error()

def generate_from_imageembeds(model, device, image_embeds, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
    prompt = [model.prompt]
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    input_ids[:, 0] = model.tokenizer.bos_token_id
    input_ids = input_ids[:, :-1]

    if sample:
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        outputs = model.text_decoder.generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_p=top_p,
            num_return_sequences=1,
            eos_token_id=model.tokenizer.sep_token_id,
            pad_token_id=model.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts)
    else:
        expanded_input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        outputs = model.text_decoder.generate(
            input_ids=expanded_input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            eos_token_id=model.tokenizer.sep_token_id,
            pad_token_id=model.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts)

    captions = [model.tokenizer.decode(output, skip_special_tokens=True)[len(model.prompt):] for output in outputs]
    return captions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--roi_str", type=str, default="early")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target", type=str, default="blip")
    parser.add_argument("--dataset", choices=["original", "occluded"], default="original")
    args = parser.parse_args()

    subject = args.subject
    roi_str = args.roi_str
    gpu = args.gpu
    target = args.target
    dataset = args.dataset

    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    image_size = 240
    tag = f"{target}_{dataset}"
    decoded_path = f"/shared/home/mis6559/neurobio240/decoded/{subject}/{subject}_{roi_str}_scores_{tag}.npy"
    savedir = f"/shared/home/mis6559/neurobio240/decoded/{subject}/captions"
    os.makedirs(savedir, exist_ok=True)

    # Load model
    print("Loading BLIP decoder...")
    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
    model_decoder = blip_decoder(
        pretrained=model_url,
        image_size=image_size,
        vit="base",
        med_config="/shared/home/mis6559/neurobio240/BLIP/configs/med_config.json"
    )
    model_decoder.eval().to(device)

    # Load features
    print(f"Loading predicted features from: {decoded_path}")
    scores = np.load(decoded_path)
    print(f" shape: {scores.shape}")

    captions_brain = []
    for imidx in tqdm(range(scores.shape[0])):
        flat = scores[imidx, :]
        if flat.shape[0] != 226 * 768:
            print(f"Skipping {imidx}: wrong shape {flat.shape}")
            continue

        scores_test = torch.tensor(flat.reshape(226, 768)).unsqueeze(0).to(device)
        caption = generate_from_imageembeds(model_decoder, device, scores_test, num_beams=3, max_length=20, min_length=5, repetition_penalty=1.5)
        captions_brain.append(caption)

    out_csv = os.path.join(savedir, f"captions_brain_{tag}.csv")
    pd.DataFrame(captions_brain).to_csv(out_csv, sep='\t', header=False, index=False)
    print(f"Saved captions to: {out_csv}")

if __name__ == "__main__":
    main()
