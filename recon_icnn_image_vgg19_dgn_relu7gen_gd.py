import sys
sys.path.insert(0, '/shared/home/mis6559/neurobio240/bdpy_clean')

import bdpy
print("Loaded bdpy from:", bdpy.__file__)

import os
import glob
import argparse
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from itertools import product
import torch
import torch.optim as optim
import PIL.Image

from bdpy.recon.torch.icnn import reconstruct
from bdpy.recon.utils import normalize_image, clip_extreme
from bdpy.dl.torch.models import VGG19, AlexNetGenerator, layer_map
from bdpy.dataform import DecodedFeatures
from bdpy.feature import normalize_feature
from bdpy.util import dump_info

def image_preprocess(img, image_mean=np.float32([104, 117, 123])):
    return np.float32(np.transpose(img, (2, 0, 1))[::-1]) - np.reshape(image_mean, (3, 1, 1))

def image_deprocess(img, image_mean=np.float32([104, 117, 123])):
    return np.dstack((img + np.reshape(image_mean, (3, 1, 1)))[::-1])

def recon_icnn_image_vgg19_dgn(subjects, layers, features_dir, output_dir, n_iter=200, device='cuda:0'):
    encoder_param_file = '/shared/home/mis6559/neurobio240/models/pytorch/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt'
    generator_param_file = '/shared/home/mis6559/neurobio240/models/pytorch/bvlc_reference_caffenet_generator_ILSVRC2012_Training/generator_relu7.pt'
    image_mean_file = '/shared/home/mis6559/neurobio240/models/pytorch/VGG_ILSVRC_19_layers/ilsvrc_2012_mean.npy'
    feature_std_file = '/shared/home/mis6559/neurobio240/models/pytorch/VGG_ILSVRC_19_layers/estimated_cnn_feat_std_VGG_ILSVRC_19_layers_ImgSize_224x224_chwise_dof1.mat'
    feature_range_file = '/shared/home/mis6559/neurobio240/models/pytorch/bvlc_reference_caffenet_generator_ILSVRC2012_Training/act_range/3x/fc7.txt'

    layer_mapping = layer_map('vgg19')
    encoder_input_shape = (224, 224, 3)
    image_mean = np.float32(np.load(image_mean_file)).mean(axis=(1, 2))
    feat_std0 = sio.loadmat(feature_std_file)
    upper_bound = np.loadtxt(feature_range_file, delimiter=' ', unpack=True).reshape((4096,))
    initial_gen_feat = np.random.normal(0, 1, (4096,))

    opts = {
        'loss_func': torch.nn.MSELoss(reduction='sum'),
        'n_iter': n_iter,
        'lr': (2., 1e-10),
        'momentum': (0.9, 0.9),
        'decay': (0.01, 0.01),
        'blurring': False,
        'channels': None,
        'masks': None,
        'disp_interval': 1,
        'initial_feature': initial_gen_feat,
        'feature_upper_bound': upper_bound,
        'feature_lower_bound': 0.,
    }

    for subject in subjects:
        save_dir = os.path.join(output_dir, subject)
        os.makedirs(save_dir, exist_ok=True)
        dump_info(save_dir, script=__file__)

        features = DecodedFeatures(features_dir, squeeze=False)
        print("Available subjects:", features.subjects)
        print("Available layers:", features.layers)
        print("Available ROIs:", features.rois)
        print("Available feature keys:")
        print("Available feature keys (raw index):")

        print("features_dir is:", features_dir)
        print("Sample query:", f'VGG19-{layers[0]}-{subject}-streams-000000')

        roi = 'streams'
        matfiles = glob.glob(os.path.join(features_dir, layers[0], subject, 'streams', '*.mat'))


        t_mat_imagenames = [os.path.splitext(os.path.basename(fl))[0] for fl in matfiles]
        print("Found matfiles:", matfiles[:5])

        print("Parsed image names:", t_mat_imagenames[:5])

        image_ids = [os.path.splitext(os.path.basename(f))[0][-6:] for f in matfiles]


        for img_id in tqdm(image_ids, desc=f"Reconstructing {subject}"):
            encoder = VGG19().to(device)
            encoder.load_state_dict(torch.load(encoder_param_file))
            encoder.eval()

            generator = AlexNetGenerator().to(device)
            generator.load_state_dict(torch.load(generator_param_file))
            generator.eval()

            snapshots_dir = os.path.join(save_dir, 'snapshots', f'image-{img_id}')
            if os.path.exists(snapshots_dir):
                print(f"Already exists: {snapshots_dir} — skipping")
                continue

            feat = {}
            for layer in layers:
                mat_path = os.path.join(features_dir, layer, subject, roi, f'VGG19-{layer}-{subject}-{roi}-{img_id}.mat')
                data = sio.loadmat(mat_path)
                raw_feat = data['feat']  # shape: (1, 64, 224, 224)
            
                norm_feat = normalize_feature(
                    raw_feat,
                    shift='self',
                    scale=np.mean(feat_std0[layer]),
                    channel_wise_mean=False,
                    channel_wise_std=False,
                    channel_axis=0,
                    std_ddof=1
                )[np.newaxis]
            
                feat[layer] = norm_feat

            feat_norm = np.array([np.linalg.norm(f) for f in feat.values()], dtype='float32')
            weights = 1. / (feat_norm ** 2)
            weights = weights / weights.sum()
            layer_weights = dict(zip(layers, weights))
            
            opts.update({'layer_weights': layer_weights})
            
            # Remove duplicate arguments just to be safe
            opts.pop('layer_mapping', None)
            opts.pop('optimizer', None)
            opts.pop('n_iter', None)
            opts.pop('loss_func', None)  # new fix
            opts['masks'] = None

            print("Keys in opts:", opts.keys())
            print("Type of layer_mapping:", type(layer_mapping))
            print("Sample from layer_mapping:", list(layer_mapping.items())[:3])

            recon_image, _ = reconstruct(
                feat,
                encoder,
                generator=generator,
                layer_mapping=layer_mapping,
                optimizer=optim.SGD,
                loss_func=torch.nn.MSELoss(reduction='sum'),
                n_iter=n_iter,
                image_size=encoder_input_shape,
                crop_generator_output=True,
                preproc=image_preprocess,
                postproc=image_deprocess,
                output_dir=save_dir,
                save_snapshot=False,
                snapshot_dir=snapshots_dir,
                snapshot_ext='tiff',
                snapshot_postprocess=normalize_image,
                return_loss=True,
                device=device,
                **opts
            )
            

            recon_image_norm = normalize_image(clip_extreme(recon_image, pct=4))
            matfile = os.path.join(save_dir, f'recon_image-{img_id}.mat')
            sio.savemat(matfile, {
                'recon_image': recon_image,
                'recon_image_norm': recon_image_norm
            })
            out_img_path = os.path.join(save_dir, f'recon_image_normalized-{img_id}.tiff')
            PIL.Image.fromarray(recon_image_norm).save(out_img_path)

if __name__ == '__main__':
    # layers = ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4', 'fc6', 'fc7', 'fc8']
    layers = ['conv1_1']
    recon_icnn_image_vgg19_dgn(
        subjects=['subj01'],
        layers=layers,
        features_dir='/shared/home/mis6559/neurobio240/decoded/gan_mod/',
        output_dir='/shared/home/mis6559/neurobio240/decoded/gan_recon_img/all_layers/',
        n_iter=200,
        device='cuda:0'
    )