# Reconstructing Occluded Images from fMRI via Latent Diffusion Models

**Author:** Miranda Shen

**Course:** NEUROBIO240 (Final Project)

**Institution:** Harvard University

## Project Overview

This project investigates whether modern generative models, specifically **latent diffusion models**, can reconstruct **partially occluded images** from **fMRI data**, simulating the human brain's capacity for **amodal completion**. Building on Takagi & Nishimoto (2023), I recreated and extended their image reconstruction pipeline to test two core hypotheses:

1. Can brain activity reveal inferred structure in **occluded visual inputs**?
2. Can the full decoding pipeline be replicated on **standard compute** using memory-optimized streaming?

Key enhancements:

* Support for **occluded image stimuli** with end-to-end decoding
* Modular reconstruction pipelines runnable on **modest hardware**
* Direct **S3 streaming of fMRI volumes** to avoid local storage bottlenecks

## Repository Structure

This project closely mirrors [yu-takagi/StableDiffusionReconstruction](https://github.com/yu-takagi/StableDiffusionReconstruction) with additional scripts and modifications for occlusion, streaming, and evaluation under resource constraints.

## File Descriptions

### fMRI Data Processing

* **`get_data.py`**
  Lists trial-wise fMRI beta weights from the NSD dataset.

* **`img2feat_sd.ipynb`**
  Downloads and caches NSD stimulus HDF5 file from S3. It aligns these data with image stimuli and saves them in a memory-mapped format for efficient access during analysis.

* **`make_subjmri.ipynb`**
  A Jupyter notebook for visualizing and debugging fMRI preprocessing steps, such as ROI masking, signal alignment, and voxel selection.

### Occlusion Pipeline

* **`generate_occluded_images.py`**
  Applies synthetic rectangular masks to NSD images to simulate occlusion, creating a dataset of partially occluded images for analysis.

### Feature Extraction (Image → Latent)

* **`img2feat_sd1_parallel.py`**
  Extracts latent representations from fully visible images using Stable Diffusion's encoder.

* **`img2feat_sd1_occlusion.py`**
  Performs the same extraction as above but on occluded images.([Scholar Friends][1])

* **`img2feat_blip.ipynb`**
  Utilizes BLIP to extract vision-language embeddings (text captions) from full images.([GitHub][2])

* **`img2feat_blip_occlusion.py`**
  Extracts BLIP embeddings from occluded images.

* **`img2feat_dpt.py`**
  Extracts depth features from images using the Dense Prediction Transformer (DPT).

* **`img2feat_decoded.py`**
  Aggregates all decoded (predicted) features for comparison or evaluation purposes.

* **`img2feat_decoded_occlusion.py`**
  Aggregates decoded features from occluded images.

### Feature-to-Brain Alignment

* **`make_subjstim.py`**
  Links each image's features to its corresponding fMRI trial using behavioral logs and NSD metadata.

* **`make_subjstim_occluded.py`**
  Performs the same linking process for occluded image stimuli.

### Regression Decoding

* **`ridge.py`**
  Trains ridge regression models to predict image features (e.g., latent vectors, captions) from fMRI data.

* **`ridge_occlusion.py`**
  Trains ridge regression models using fMRI data mapped to occluded images.

### Image Reconstruction

* **`diffusion_decoding.py`**
  Reconstructs images from decoded features using Stable Diffusion. It supports different modes:

  * `--method latent` (default)
  * `--method text` (text prompt from BLIP)
  * `--method gan` (refined via GAN)

* **`diffusion_decoding_occluded.py`**
  Performs image reconstruction from decoded features of occluded images.

* **`decode_captions.ipynb`**
  A Jupyter notebook for decoding captions from fMRI data using trained models.

* **`decode_captions_occluded.py`**
  Decodes captions from fMRI data corresponding to occluded images.

* **`dptemb2dpt.py`**
  Converts DPT embeddings back to depth maps for visualization or further processing.

* **`combine_roi_predictions.py`**
  Combines predictions from different regions of interest (ROIs) to improve overall decoding performance.

### Evaluation

* **`identification.py`**
  Evaluates how well reconstructed images match original stimuli using cosine similarity in feature space (e.g., Inception v3 or CLIP embeddings).

* **`identification_occluded.py`**
  Evaluates reconstructions from occluded image decoding using similar metrics.

## MRI Preprocessing

### `get_data.py`

**List NSD subject files from S3.**

```bash
python get_data.py ppdata subj01
```
### `make_subjmri.ipynb`

**Extract fMRI data and behavioral responses for a subject.**

```bash
jupyter notebook make_subjmri.ipynb
```

Or run as script:

```bash
jupyter nbconvert --to script make_subjmri.ipynb
python make_subjmri.py
```
## Reconstruction Based on CVPR Method

### `img2feat_sd.ipynb`

**Download and cache NSD stimulus HDF5 file from S3.**

```bash
jupyter notebook img2feat_sd.ipynb
```

### `make_subjstim.py`

**Generate feature matrices from image embeddings per subject.**

```bash
python make_subjstim.py --subject subj01 --featname blip --use_stim ave
```

### `ridge.py`

**Train ridge regression models to predict features from fMRI data.**

```bash
python ridge.py --subject subj01 --target blip --roi early --use_stim each
```

### `diffusion_decoding.py`

**Reconstruct images from predicted features using Stable Diffusion.**

```bash
python diffusion_decoding.py --imgidx 0 10 --gpu 1 --subject subj01 --method cvpr
```

## Reconstruction with Decoded Text Prompt

### `img2feat_blip.ipynb`

**Extract BLIP text features from stimulus images.**

```bash
jupyter notebook img2feat_blip.ipynb
```

### `make_subjstim.py`

**Create subject-specific stimulus feature matrices (BLIP text features).**

```bash
python make_subjstim.py --featname blip --use_stim ave --subject subj01
python make_subjstim.py --featname blip --use_stim each --subject subj01
```

### `ridge.py`

**Train regression models from fMRI to text feature space.**

```bash
python ridge.py --target blip --roi early ventral midventral midlateral lateral parietal --subject subj01
```

### `decode_captions.ipynb`

**Generate decoded text captions from predicted features.**

```bash
jupyter notebook decode_captions.ipynb
```

> If converted to script:

```bash
python decode_captions.py --subject subj01
```

### `diffusion_decoding.py`

**Reconstruct images from decoded text using Stable Diffusion.**

```bash
python diffusion_decoding.py --imgidx 0 --gpu 1 --subject subj01 --method text
```
## Reconstruction with Decoded Text Prompt + GAN

### `make_vgg19bdpy.py`

**Extract VGG19 features for NSD stimulus images.**

```bash
python make_vgg19bdpy.py --imgidx 0 73000
```

> You can adjust the image index range (e.g., `--imgidx 0 1000`) to limit processing.

### `make_subjstim_vgg19.py`

**Create feature matrices for each VGG19 layer.**

```bash
python make_subjstim_vgg19.py --layer conv1_1 --subject subj01
python make_subjstim_vgg19.py --layer conv1_2 --subject subj01
...
python make_subjstim_vgg19.py --layer fc8 --subject subj01
```

### `ridge.py`

**Train regression models from fMRI to VGG19 feature layers.**

```bash
python ridge.py --target conv1_1 --roi early ventral midventral midlateral lateral parietal --subject subj01
```

### `make_vgg19fromdecode.py`

**Combine decoded VGG19 features from multiple ROIs.**

```bash
python make_vgg19fromdecode.py --subject subj01
```

### `recon_icnn_image_vgg19_dgn_relu7gen_gd.py`

**Reconstruct images from decoded VGG19 features using GAN inversion.**

```bash
python recon_icnn_image_vgg19_dgn_relu7gen_gd.py
```

### `diffusion_decoding.py`

**Use the GAN reconstruction as conditioning input to Stable Diffusion.**

```bash
python diffusion_decoding.py --imgidx 0 --gpu 0 --subject subj01 --method gan
```

## Reconstruction with Decoded Text Prompt + GAN + Decoded Depth

> **Note:** Requires updating `transformers`. This may break BLIP compatibility. Use a separate virtual environment.

```bash
pip install -U transformers
```

### `img2feat_dpt.py`

**Extract DPT depth embeddings from stimulus images.**

```bash
python img2feat_dpt.py --imgidx 0 73000 --gpu 0
```

### `make_subjstim.py`

**Create stimulus feature matrices for DPT embeddings.**

```bash
python make_subjstim.py --featname dpt_emb0 --use_stim ave --subject subj01
python make_subjstim.py --featname dpt_emb0 --use_stim each --subject subj01
```

Repeat for `dpt_emb1`, `dpt_emb2`, `dpt_emb3` if needed.

### `ridge.py`

**Train models to decode depth features from fMRI.**

```bash
python ridge.py --target dpt_emb0 --roi early ventral midventral midlateral lateral parietal --subject subj01
```

Repeat for each DPT embedding layer.

### `dptemb2dpt.py`

**Convert decoded DPT embeddings into full-resolution depth maps.**

```bash
python dptemb2dpt.py --gpu 0 --subject subj01
```

### `diffusion_decoding.py`

**Fuse text, GAN, and depth priors for final image reconstruction.**

```bash
python diffusion_decoding.py --imgidxs 0 1 --gpu 0 --subject subj01
```

## Occluded Image-to-Brain-to-Image Reconstruction

### 

**Generate occluded stimulus images.**

If starting from clean images:

```bash
python generate_occluded_images.py --input_path /path/to/nsd_stimuli.hdf5 --output_path ./nsd_stimuli_occluded.hdf5
```

### `img2feat_sd1_occlusion.py`

**Extract features from occluded images.**

```bash
python img2feat_sd1_occlusion.py --imgidx 0 73000 --gpu 0 --dataset occluded
```

### `make_subjstim_occluded.py`

**Create fMRI feature matrices.**

```bash
python make_subjstim_occluded.py --featname init_latent_occluded --use_stim each --subject subj01 --input_dir ./nsdfeat/init_latent_occluded
python make_subjstim_occluded.py --featname init_latent_occluded --use_stim ave  --subject subj01 --input_dir ./nsdfeat/init_latent_occluded

python make_subjstim_occluded.py --featname c_occluded --use_stim each --subject subj01 --input_dir ./nsdfeat/c
python make_subjstim_occluded.py --featname c_occluded --use_stim ave  --subject subj01 --input_dir ./nsdfeat/c
```

### `ridge_occlusion.py`

**Decode from fMRI.**

```bash
python ridge_occlusion.py --target c_occluded --roi ventral --subject subj01 --use_stim each --dataset occluded
python ridge_occlusion.py --target init_latent_occluded --roi early --subject subj01 --use_stim each --dataset occluded
```

### `decode_captions_occluded.py`

**Decode captions.**

```bash
python decode_captions_occluded.py --subject subj01 --target init_latent_occluded --roi_str early --dataset occluded --gpu 0
```

### `diffusion_decoding_occluded.py`

**Reconstruct images.**

```bash
python diffusion_decoding_occluded.py --imgidx 0 --gpu 0 --subject subj01 --method cvpr
```

### `img2feat_decoded_occlusion.py`

**Extract features from reconstructions.**

```bash
python img2feat_decoded_occlusion.py --gpu 0 --subject subj01 --method cvpr
```

### `identification_occluded.py`

**Evaluate reconstruction accuracy.**

```bash
python identification_occluded.py --usefeat inception --subject subj01 --method cvpr
```

Try also:

```bash
--usefeat alexnet12
--usefeat clip
--usefeat clip_h12
```

## Acknowledgments

This project builds on:

* **Takagi & Nishimoto (2023)** – [StableDiffusionReconstruction](https://github.com/yu-takagi/StableDiffusionReconstruction)
* **NSD Dataset** – Natural Scenes Dataset
* **Pretrained Models Used**:

  * Stable Diffusion v1.4
  * BLIP (Salesforce)
  * VGG19 (ImageNet)
  * DPT (HuggingFace)
  * Inception v3 (PyTorch)

Special thanks to:

* ChatGPT for debugging assistance
* The NSD community for data/tools
* My NEUROBIO240 instructors and peers
