# VQ-VAE for Image and Speech Representation Learning

This project implements Vector Quantized Variational Autoencoders (VQ-VAE) for unsupervised representation learning on image and speech data, as part of a Master's thesis at Rutgers University (Spring 2025). The project applies VQ-VAE to the CIFAR-10 dataset for image reconstruction and the L2-ARCTIC dataset for mel spectrogram reconstruction, focusing on learning compact, discrete latent representations. The implementation is in PyTorch, using Jupyter notebooks for interactive experimentation.

## Repository Structure
```
VQVAE-main/
├── Image Data/
│   ├── loss_vs_epochs.png
│   ├── original_and_reconstructed.png
│   └── VQVAE_Image.ipynb
├── Speech Data/
│   ├── VQVAE_Speech.ipynb
│   ├── L2-Subset.ipynb
│   ├── speech_reconstruction.png
│   └── speech_loss.png
└── README.md
```

- **Image Data/VQVAE_Image.ipynb**: Notebook implementing VQ-VAE for image reconstruction on CIFAR-10.
- **Image Data/loss_vs_epochs.png**: Plot of training loss curves (reconstruction, VQ, and total) for CIFAR-10.
- **Image Data/original_and_reconstructed.png**: Comparison of original and reconstructed images from CIFAR-10.
- **Speech Data/VQVAE_Speech.ipynb**: Notebook implementing VQ-VAE for mel spectrogram reconstruction on L2-ARCTIC.
- **Speech Data/L2-Subset.ipynb**: Notebook to subset the L2-ARCTIC dataset (selecting speakers `HJK`, `BWC`, `YBAA`, `SVBI`, `THV` and limiting to 500 files per speaker).
- **Speech Data/speech_reconstruction.png**: Comparison of original and reconstructed mel spectrograms from L2-ARCTIC.
- **Speech Data/speech_loss.png**: Plot of training loss curves (reconstruction, VQ, and total) for L2-ARCTIC.
- **README.md**: This file, describing the project and usage.

## Project Description
This project explores unsupervised representation learning using Vector Quantized Variational Autoencoders (VQ-VAE) for image and speech data. VQ-VAE introduces a discrete latent space to learn compact, symbolic representations, improving interpretability and avoiding issues like posterior collapse in traditional VAEs. The model is applied to:
- **Images**: CIFAR-10 dataset, reconstructing 32x32 color images.
- **Speech**: L2-ARCTIC dataset, reconstructing mel spectrograms for speaker-independent accent and phonetic content.

The VQ-VAE consists of an encoder (convolutional layers with residual blocks), a vector quantizer (discrete codebook with 512 embeddings, β=0.25), and a decoder (transposed convolutions). The model is trained end-to-end with a loss combining reconstruction (MSE) and quantization terms, using the straight-through estimator for backpropagation through the non-differentiable quantization step.

## Problem Statement
The goal is to learn efficient, unsupervised representations of high-dimensional data (images and speech) using VQ-VAE. The model maps input data \( x \) to a discrete latent representation \( z \), retaining sufficient information for high-quality reconstruction while enabling compression and generative modeling. The problem is framed as maximizing the likelihood \( p(x) = \sum_z p(x|z)p(z) \), optimized via the variational lower bound (ELBO):

\[
\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}[q(z|x) || p(z)]
\]

Key challenges include capturing essential structures in visual and auditory data, ensuring discrete latent codes are expressive, and achieving high-fidelity reconstructions. The project evaluates performance through reconstruction loss, codebook utilization (perplexity), and visual/audio quality.

## Prerequisites
- Python 3.8+
- PyTorch (with CUDA for GPU support)
- Librosa
- NumPy
- Matplotlib
- Scikit-learn
- TQDM
- Jupyter Notebook or Google Colab
- Datasets:
  - CIFAR-10 (auto-downloaded via PyTorch)
  - L2-ARCTIC (manually downloaded)

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/VQVAE-main.git
cd VQVAE-main
```

### 2. Install Dependencies
Create a virtual environment and install required packages:
```bash
python -m venv vqvae_env
source vqvae_env/bin/activate  # Linux/Mac
vqvae_env\Scripts\activate    # Windows
pip install torch torchaudio librosa numpy matplotlib scikit-learn tqdm jupyter
```

### 3. Prepare the Datasets
- **CIFAR-10** (Images):
  - Automatically downloaded by `Image Data/VQVAE_Image.ipynb`.
- **L2-ARCTIC** (Speech):
  - Download from [CMU L2-ARCTIC](https://www.cs.cmu.edu/~awb/l2_arctic/).
  - Organize in:
    ```
    L2_ARCTIC_SUBSET/
    ├── HJK/
    │   └── wav/
    ├── BWC/
    ├── YBAA/
    ├── SVBI/
    ├── THV/
    ```
  - Update `DATASET_PATH` in `Speech Data/VQVAE_Speech.ipynb` and `Speech Data/L2-Subset.ipynb` (e.g., `./L2_ARCTIC_SUBSET` or `/content/drive/MyDrive/Colab Notebooks/L2_ARCTIC_SUBSET` for Colab).
  - Update `MEL_CACHE_DIR` in `Speech Data/VQVAE_Speech.ipynb` (e.g., `./L2_ARCTIC_MEL_CACHE`).

### 4. Configure Google Colab (Optional)
If running in Colab:
- Upload the repository:
  ```bash
  !git clone https://github.com/your-username/VQVAE-main.git
  %cd VQVAE-main
  ```
- Install dependencies:
  ```bash
  !pip install torch librosa numpy matplotlib scikit-learn tqdm
  ```
- Mount Google Drive:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- Update dataset paths in notebooks as needed.

## Usage
1. **Prepare the L2-ARCTIC Subset**:
   - Open `Speech Data/L2-Subset.ipynb` in Jupyter or Colab.
   - Run cells to create the dataset subset (selects speakers `HJK`, `BWC`, `YBAA`, `SVBI`, `THV`; limits to 500 files per speaker).
   - Ensure `DATASET_PATH` points to your L2-ARCTIC dataset.

2. **Run the Speech VQ-VAE**:
   - Open `Speech Data/VQVAE_Speech.ipynb` in Jupyter or Colab.
   - Run cells to:
     - Cache mel spectrograms in `MEL_CACHE_DIR`.
     - Train the VQ-VAE for 50 epochs (batch size 16, Adam optimizer, learning rate 2e-4).
     - Plot loss curves (saved as `speech_loss.png`).
     - Evaluate on a validation sample, displaying spectrograms (saved as `speech_reconstruction.png`) and playing audio.
   - Outputs are saved to `Speech Data/`.

3. **Run the Image VQ-VAE**:
   - Open `Image Data/VQVAE_Image.ipynb` in Jupyter or Colab.
   - Run cells to:
     - Train the VQ-VAE on CIFAR-10.
     - Plot loss curves (saved as `loss_vs_epochs.png`).
     - Display reconstructed images (saved as `original_and_reconstructed.png`).

## Results
The VQ-VAE achieves strong reconstruction quality for both modalities, with discrete latent representations capturing high-level semantics.

- **Images (CIFAR-10)**:
  - **Loss Curves**: [Image Data/loss_vs_epochs.png](Image%20Data/loss_vs_epochs.png) shows steady decreases in reconstruction and VQ losses, indicating effective learning.
  - **Reconstruction**: [Image Data/original_and_reconstructed.png](Image%20Data/original_and_reconstructed.png) demonstrates preserved global structure and class identity, though fine details are smoothed due to MSE loss.

- **Speech (L2-ARCTIC)**:
  - **Loss Curves**: [Speech Data/speech_loss.png](Speech%20Data/speech_loss.png) shows consistent reduction in reconstruction loss, with stable VQ loss, confirming meaningful discrete audio representations.
  - **Reconstruction**: [Speech Data/speech_reconstruction.png](Speech%20Data/speech_reconstruction.png) illustrates close matching of original and reconstructed mel spectrograms, preserving energy and phonetic structure. Audio remains intelligible despite Griffin-Lim synthesis artifacts.

Quantitative metrics include reconstruction loss (MSE), codebook utilization (perplexity), and visual inspection of spectrogram quality (formant structures, syllabic boundaries).

## Future Work
- **Speaker Variation**: The current speech VQ-VAE focuses on mel spectrogram reconstruction. Work is in progress to incorporate speaker variation in the decoder (e.g., via speaker embeddings for accent conversion), to be added in a future update.
- **Perceptual Losses**: Integrate perceptual or adversarial losses (e.g., STFT, GAN-based) to improve fine details in image and audio reconstructions.
- **Latent Prior Learning**: Add a WaveNet or Transformer-based prior over discrete codes to enable generative sampling.
- **High-Quality Vocoder**: Replace Griffin-Lim with a vocoder like HiFi-GAN for better speech synthesis.
- **Quantitative Metrics**: Include additional metrics (e.g., PESQ, MOS) for speech quality evaluation.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, features, or improvements.

## Licensing
This project is not currently licensed. Please contact the author for permission to use or distribute.

## Acknowledgements
This project was conducted under the supervision of Prof. Gemma Moran at Rutgers University, whose guidance was instrumental. Thanks to peers for discussions and family for support.