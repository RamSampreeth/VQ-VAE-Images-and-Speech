# VQ-VAE for Image and Speech Representation Learning

This project implements Vector Quantized Variational Autoencoders (VQ-VAE) for unsupervised representation learning on image and speech data, as part of a Master's thesis at Rutgers University (Spring 2025). The project applies VQ-VAE to the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) for image reconstruction and the [L2-ARCTIC dataset](https://psi.engr.tamu.edu/l2-arctic-corpus/) for mel spectrogram reconstruction, focusing on learning compact, discrete latent representations. The implementation is in PyTorch, using Jupyter notebooks for interactive experimentation.

---

## 📁 Repository Structure

```
VQVAE-main/
├── Image Data/
│   ├── loss_vs_epochs.png
│   ├── original_and_reconstructed.png
│   └── VQVAE_Image.ipynb
├── Speech Data/
│   ├── VQVAE_Speech.ipynb
│   ├── L2-Subset.ipynb
│   ├── speech_loss.png
│   └── speech_reconstruction.png
└── README.md
```

- **VQVAE_Image.ipynb**: Trains and evaluates VQ-VAE on CIFAR-10 images.
- **[loss_vs_epochs.png](Image%20Data/loss_vs_epochs.png)**: Training loss curves for the image model.
- **[original_and_reconstructed.png](Image%20Data/original_and_reconstructed.png)**: Reconstructed CIFAR-10 image samples.
- **VQVAE_Speech.ipynb**: VQ-VAE model for mel spectrogram reconstruction on L2-ARCTIC.
- **L2-Subset.ipynb**: Subsets the L2-ARCTIC dataset (for user-selected speakers, up to 500 utterances each).
- **[speech_loss.png](Speech%20Data/speech_loss.png)**: Speech model loss curves.
- **[speech_reconstruction.png](Speech%20Data/speech_reconstruction.png)**: Original vs. reconstructed mel spectrograms.

---

## 🧠 Project Overview

VQ-VAE introduces a discrete latent space for unsupervised representation learning, improving interpretability and eliminating issues like posterior collapse in traditional VAEs. The architecture consists of:

- A convolutional **encoder** with residual blocks
- A **vector quantizer** (512-codebook embeddings, β = 0.25)
- A transposed convolution **decoder**

The loss combines MSE reconstruction, commitment, and embedding terms, trained with the straight-through estimator for backpropagation through the non-differentiable quantization step.

---

## 🔧 Problem Statement

The model maps input data \( x \) to discrete latent codes \( z \), enabling efficient reconstruction while allowing compression and symbolic representation learning:

\[
\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}[q(z|x) \| p(z)]
\]

**Challenges addressed**:
- Capturing meaningful structure in images and speech
- Learning expressive discrete embeddings
- Achieving high-quality reconstruction

The project evaluates performance through reconstruction loss (MSE), codebook utilization (perplexity), and visual/audio quality.

---

## 🛠️ Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/) (with CUDA if available)
- [Librosa](https://librosa.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [tqdm](https://tqdm.github.io/)
- [Jupyter](https://jupyter.org/) or [Google Colab](https://colab.research.google.com/)

**Install dependencies**:
```bash
pip install torch torchaudio librosa numpy matplotlib scikit-learn tqdm jupyter
```

---

## 📦 Datasets

### CIFAR-10 (Images)
- Auto-downloaded via `torchvision.datasets` in `VQVAE_Image.ipynb`.
- 60,000 color images (32x32) across 10 classes.

### L2-ARCTIC (Speech)
- **Important**: Obtain access to the [L2-ARCTIC dataset](https://psi.engr.tamu.edu/l2-arctic-corpus/) before running the speech code. Select the speakers you want to use (e.g., `HJK`, `BWC`, `YBAA`, `SVBI`, `THV`).
- Download from the official source and organize in:
  ```
  L2_ARCTIC_SUBSET/
  ├── <Speaker1>/wav/
  ├── <Speaker2>/wav/
  └── ...
  ```
- Update paths in `VQVAE_Speech.ipynb` and `L2-Subset.ipynb`:
  - `DATASET_PATH` (e.g., `./L2_ARCTIC_SUBSET` or `/content/drive/MyDrive/Colab Notebooks/L2_ARCTIC_SUBSET`)
  - `MEL_CACHE_DIR` (e.g., `./L2_ARCTIC_MEL_CACHE`)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/VQVAE-main.git
cd VQVAE-main
```

### 2. Prepare the L2-ARCTIC Subset
- Open `Speech Data/L2-Subset.ipynb` in Jupyter or Colab.
- Run cells to:
  - Specify your selected speakers
  - Limit to 500 utterances per speaker (adjustable)
  - Generate metadata and spectrogram cache

### 3. Run the Speech Model
- Open `Speech Data/VQVAE_Speech.ipynb` in Jupyter or Colab.
- Run cells to:
  - Cache mel spectrograms in `MEL_CACHE_DIR`
  - Train VQ-VAE (50 epochs, batch size 16, learning rate 2e-4)
  - Save loss plots (`speech_loss.png`)
  - Evaluate spectrograms and audio (`speech_reconstruction.png`)

### 4. Run the Image Model
- Open `Image Data/VQVAE_Image.ipynb` in Jupyter or Colab.
- Run cells to:
  - Train VQ-VAE on CIFAR-10
  - Save loss plots (`loss_vs_epochs.png`)
  - Reconstruct and visualize images (`original_and_reconstructed.png`)

### 5. Configure Google Colab (Optional)
- Upload repository:
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

---

## 📈 Results

### Images (CIFAR-10)
- **Loss Curves**: [loss_vs_epochs.png](Image%20Data/loss_vs_epochs.png) shows smooth convergence for reconstruction and VQ loss, indicating effective learning.
- **Reconstruction**: [original_and_reconstructed.png](Image%20Data/original_and_reconstructed.png) preserves global structure and class identity, with minor blurring due to MSE loss.

### Speech (L2-ARCTIC)
- **Loss Curves**: [speech_loss.png](Speech%20Data/speech_loss.png) demonstrates consistent reduction in reconstruction loss and stable VQ loss, confirming meaningful discrete audio representations.
- **Reconstruction**: [speech_reconstruction.png](Speech%20Data/speech_reconstruction.png) shows close matching of original and reconstructed mel spectrograms, retaining phonetic detail and intelligibility despite Griffin-Lim synthesis artifacts.

**Quantitative Metrics**:
- Reconstruction loss (MSE)
- Codebook utilization (perplexity)
- Visual inspection of spectrogram quality (formant structures, syllabic boundaries)

---

## 🔮 Future Work

- **Speaker Variation**: Work is in progress to incorporate speaker variation in the speech VQ-VAE decoder (e.g., using speaker embeddings for accent conversion), to be added in a future update.
- **Perceptual Losses**: Integrate STFT or adversarial losses (e.g., GAN-based) to enhance fine details in reconstructions.
- **Latent Prior Modeling**: Add a WaveNet or Transformer prior for generative sampling from the latent space.
- **High-Quality Vocoder**: Replace Griffin-Lim with [HiFi-GAN](https://arxiv.org/abs/2010.05646) or [Parallel WaveGAN](https://arxiv.org/abs/1910.11480) for improved speech synthesis.
- **Quantitative Evaluation**: Include metrics like [PESQ](https://ieeexplore.ieee.org/document/9052948), [STOI](https://ieeexplore.ieee.org/document/5952137), or MOS for audio quality.

---

## 🤝 Contributing

Contributions are welcome! Open an [issue](https://github.com/your-username/VQVAE-main/issues) or submit a pull request for:
- Feature enhancements
- Bug fixes
- Additional dataset support

---

## 🧾 Licensing

This project is currently not licensed. Please contact the author for use or distribution rights.

---

## 🔗 References

- [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
- [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)
- [L2-ARCTIC Corpus](https://psi.engr.tamu.edu/l2-arctic-corpus/)
- [librosa: Audio and Music Signal Analysis in Python](https://librosa.org/)