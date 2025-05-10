import argparse, os, glob
from PIL import Image
import torch
import numpy as np
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--subject", required=True, type=str)
    parser.add_argument("--method", required=True, type=str)  # usually 'cvpr'
    args = parser.parse_args()

    method = args.method
    subject = args.subject
    gpu = args.gpu

    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    input_dir = f'/shared/home/mis6559/neurobio240/decoded/image-{method}-occluded/{subject}/samples/'
    outdir = f'/shared/home/mis6559/neurobio240/identification/{method}_occluded/{subject}/'
    os.makedirs(outdir, exist_ok=True)
    imglist = sorted(glob.glob(f'{input_dir}/*.png'))

    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Inception
    model_inception = models.inception_v3(pretrained=True)
    model_inception.eval().to(device)
    model_inception = create_feature_extractor(model_inception, {'flatten': 'flatten'})

    # AlexNet
    model_alexnet = models.alexnet(pretrained=True)
    model_alexnet.eval().to(device)
    model_alexnet = create_feature_extractor(model_alexnet, {
        'features.5': 'features.5',
        'features.12': 'features.12',
        'classifier.5': 'classifier.5'
    })

    # CLIP
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    print(f"Extracting features for occluded {method}/{subject} ...")

    for img_path in tqdm(imglist):
        imgname = os.path.splitext(os.path.basename(img_path))[0]
        image = Image.open(img_path).convert("RGB")

        input_tensor = preprocess(image).unsqueeze(0).to(device)

        # Inception
        with torch.no_grad():
            feat_inception = model_inception(input_tensor)['flatten'].cpu().numpy()

        # AlexNet
        with torch.no_grad():
            feats = model_alexnet(input_tensor)
        feat_alexnet5 = feats['features.5'].flatten().cpu().numpy()
        feat_alexnet12 = feats['features.12'].flatten().cpu().numpy()
        feat_alexnet18 = feats['classifier.5'].flatten().cpu().numpy()

        # CLIP
        inputs = processor_clip(text="", images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_clip(**inputs, output_hidden_states=True)
        feat_clip = outputs.image_embeds.cpu().numpy()
        feat_clip_h6 = outputs.vision_model_output.hidden_states[6].flatten().cpu().numpy()
        feat_clip_h12 = outputs.vision_model_output.hidden_states[12].flatten().cpu().numpy()

        # Save
        fname = os.path.join(outdir, imgname)
        np.save(f'{fname}_inception.npy', feat_inception)
        np.save(f'{fname}_alexnet5.npy', feat_alexnet5)
        np.save(f'{fname}_alexnet12.npy', feat_alexnet12)
        np.save(f'{fname}_alexnet18.npy', feat_alexnet18)
        np.save(f'{fname}_clip.npy', feat_clip)
        np.save(f'{fname}_clip_h6.npy', feat_clip_h6)
        np.save(f'{fname}_clip_h12.npy', feat_clip_h12)

if __name__ == "__main__":
    main()
