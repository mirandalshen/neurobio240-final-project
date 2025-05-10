import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.io import savemat
from torchvision.models import vgg19
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import h5py

class ImageDatasetNSD(Dataset):
    def __init__(self, resize=(224, 224), shape='chw', transform=None, scale=255, rgb_mean=(123.68, 116.779, 103.939), nsd_dir='../../nsd/'):
        self.h5_path = os.path.join(nsd_dir, '/shared/home/mis6559/neurobio240/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
        self.resize = resize
        self.shape = shape
        self.transform = transform
        self.scale = scale
        self.rgb_mean = rgb_mean
        self.file = None  # Lazy-opened HDF5 file

        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['imgBrick'].shape[0]

        self.default_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
        img = self.file['imgBrick'][index]

        img = Image.fromarray(img.astype(np.uint8))
        img = self.default_transform(img) * self.scale

        if self.shape == 'chw':
            img = img[[2, 1, 0], :, :]  # RGB to BGR

        img = img - torch.tensor(self.rgb_mean).view(3, 1, 1)
        return img.float()

    def __len__(self):
        return self.length

    def __del__(self):
        if self.file is not None:
            self.file.close()

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self.activations = {}
        for name in layers:
            module = self._get_submodule(name)
            module.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def fn(_, __, output):
            self.activations[name] = output.detach().cpu().numpy()
        return fn

    def _get_submodule(self, name):
        parts = name.replace('[', '.').replace(']', '').split('.')
        mod = self.model
        for part in parts:
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        return mod

    def forward(self, x):
        self.activations = {}
        _ = self.model(x)
        return self.activations

def main():
    import sys
    imgidx = list(map(int, sys.argv[1:3]))
    start, end = imgidx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    encoder_param_file = './models/pytorch/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt'
    image_mean_file = './models/pytorch/VGG_ILSVRC_19_layers/ilsvrc_2012_mean.npy'
    image_mean = np.load(image_mean_file)
    image_mean = np.float32([image_mean[0].mean(), image_mean[1].mean(), image_mean[2].mean()])

    encoder = vgg19()
    encoder.load_state_dict(torch.load(encoder_param_file))
    encoder.to(device)
    encoder.eval()

    maps = {
        'conv1_1': 'features[0]', 'conv1_2': 'features[2]',
        'conv2_1': 'features[5]', 'conv2_2': 'features[7]',
        'conv3_1': 'features[10]', 'conv3_2': 'features[12]', 'conv3_3': 'features[14]', 'conv3_4': 'features[16]',
        'conv4_1': 'features[19]', 'conv4_2': 'features[21]', 'conv4_3': 'features[23]', 'conv4_4': 'features[25]',
        'conv5_1': 'features[28]', 'conv5_2': 'features[30]', 'conv5_3': 'features[32]', 'conv5_4': 'features[34]',
        'fc6': 'classifier[0]', 'fc7': 'classifier[3]', 'fc8': 'classifier[6]'
    }

    fext = FeatureExtractor(encoder, layers=list(maps.values()))
    outdir = '/shared/home/mis6559/neurobio240/nsdfeat/vgg19_features/'

    # Pre-create all output directories
    for key in maps:
        os.makedirs(f'{outdir}/{key}/nsd/org/', exist_ok=True)

    imgdata = ImageDatasetNSD(resize=(224, 224), shape='chw', transform=None, scale=255, rgb_mean=image_mean, nsd_dir='../../nsd/')
    subset = Subset(imgdata, range(start, end))
    loader = DataLoader(subset, batch_size=8, num_workers=4, pin_memory=True)

    for batch_idx, imgs in enumerate(tqdm(loader)):
        idx_base = start + batch_idx * imgs.size(0)
        with torch.no_grad():
            batch = imgs.to(device)
            feat = fext(batch)

        for b_idx in range(batch.size(0)):
            for key, layer in maps.items():
                save_path = f'{outdir}/{key}/nsd/org/VGG19-{key}-nsd-org-{idx_base + b_idx:06}.mat'
                if os.path.exists(save_path):
                    continue  # Already done
                cfeat = feat[layer][b_idx].copy().squeeze()
                savemat(save_path, {"feat": cfeat})
        

if __name__ == "__main__":
    main()
