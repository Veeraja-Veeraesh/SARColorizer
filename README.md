# SARColorizer
# 🛰️ Advanced Satellite Image Colorization: SAR to Optical 

This project implements a deep learning model for colorizing grayscale Synthetic Aperture Radar (SAR) images (Sentinel-1) into realistic optical-like color images (Sentinel-2). The core of the model is a CycleGAN-enhanced Dual Pix2Pix network, further improved by incorporating VGG-based perceptual content loss. The entire pipeline is built using PyTorch and includes Weights & Biases (W&B) integration for robust experiment tracking.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![wandb](https://img.shields.io/badge/Weights%20&%20Biases-Tracked-blue)](https://wandb.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Overview

The primary goal is to translate images from a grayscale SAR domain (Domain A: Sentinel-1) to a color optical domain (Domain B: Sentinel-2). This is a challenging image-to-image translation task due to the significant domain gap between SAR and optical imagery.

The model leverages:
*   **CycleGAN Framework:** To learn mappings in both directions (A → B and B → A) and enforce cycle consistency, enabling training with unpaired data if necessary (though this project uses paired data).
*   **Dual Pix2Pix Architecture:** Two U-Net based generators and two PatchGAN discriminators form the core of the CycleGAN.
    *   `G_AB`: Primary generator for SAR grayscale (A) to Optical color (B).
    *   `G_BA`: Generator for Optical color (B) to SAR grayscale (A).
    *   `D_Y`: Discriminator for distinguishing real optical images from `G_AB(A)`.
    *   `D_X`: Discriminator for distinguishing real SAR images from `G_BA(B)`.
*   **VGG16 Content Loss:** A pre-trained VGG16 network is used to compute perceptual loss (content loss) between the generated color image and the target color image. This helps ensure the colorized images retain structural and semantic fidelity.
*   **Paired Data L1 Loss:** For the primary `G_AB` generator, a direct L1 (Mean Absolute Error) loss is applied against the ground truth color images, as paired data is available.

---

## ✨ Key Features

*   **Sophisticated Architecture:** Combines the strengths of CycleGAN and Pix2Pix with perceptual loss.
*   **U-Net Generator:** Employs a U-Net architecture with skip connections for the generators, effective at preserving spatial information.
*   **PatchGAN Discriminator:** Uses PatchGAN discriminators that classify N x N patches of an image as real or fake, encouraging sharper and more detailed outputs.
*   **Perceptual Fidelity:** VGG16-based content loss ensures high-level feature similarity.
*   **Dataset:** Trained on paired Sentinel-1 (SAR) and Sentinel-2 (Optical) images from the [Sentinel12 Image Pairs Segregated by Terrain](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain) Kaggle dataset.
    *   Currently, the notebook is configured to use a subset of images (`[:1000]` per terrain class in `SentinelDataset`) for faster experimentation.
*   **PyTorch Implementation:** Fully implemented in PyTorch.
*   **Weights & Biases Integration 📊:**
    *   Logs hyperparameters, losses (batch-wise and epoch-wise averages).
    *   Saves example generated images periodically.
    *   Stores model checkpoints as W&B Artifacts.
*   **Kaggle Ready 🚀:** Designed to run smoothly in a Kaggle Notebook environment with GPU acceleration.
*   **Memory Optimization:** Includes techniques like `torch.cuda.empty_cache()`, `gc.collect()`, and careful tensor management for training on GPUs with limited VRAM (e.g., 16GB P100).

---

## 🏗️ Model Architecture Deep Dive

1.  **Generators (`G_AB` and `G_BA`):**
    *   **Type:** U-Net architecture.
    *   **Input:**
        *   `G_AB`: Grayscale image (1 channel, e.g., 256x256x1).
        *   `G_BA`: Color image (3 channels, e.g., 256x256x3).
    *   **Output:**
        *   `G_AB`: Colorized image (3 channels, e.g., 256x256x3).
        *   `G_BA`: Grayscale image (1 channel, e.g., 256x256x1).
    *   **Details:** Encoder-decoder structure with skip connections. Uses LeakyReLU in the encoder and ReLU in the decoder. Tanh activation for the final output to scale images to [-1, 1].

2.  **Discriminators (`D_X` and `D_Y`):**
    *   **Type:** PatchGAN.
    *   **Input:** Concatenation of the source image and the (real or generated) target image.
        *   `D_Y`: Concatenated (Grayscale Source, Real/Fake Color Target).
        *   `D_X`: Concatenated (Color Source, Real/Fake Grayscale Target).
    *   **Output:** A 2D feature map where each "pixel" classifies a patch of the input (real vs. fake). No sigmoid in the final layer, as `BCEWithLogitsLoss` is used.
    *   **Details:** Series of convolutional layers with BatchNorm (except the first) and LeakyReLU activations.

3.  **VGG16 Content Loss Network:**
    *   A pre-trained VGG16 model (frozen, not trained).
    *   Features are extracted from an intermediate convolutional layer (e.g., `block3_conv3` / `relu3_3`).
    *   The L1 distance between the VGG features of the generated color image and the real color image constitutes the content loss.
    *   Input images are denormalized from [-1, 1] to [0, 1] and then normalized according to VGG's standard ImageNet mean/std before passing through VGG.

4.  **Loss Functions:**
    *   **Adversarial Loss (GAN Loss):** `BCEWithLogitsLoss` for both generators and discriminators.
    *   **Cycle Consistency Loss:** L1 norm ensuring `F(G(X)) ≈ X` and `G(F(Y)) ≈ Y`. Weighted by `LAMBDA_CYCLE`.
    *   **L1 (MAE) Loss for `G_AB`:** Direct L1 penalty between `G_AB(X)` and the ground truth color image `Y_target`. Weighted by `LAMBDA_L1`.
    *   **VGG Content Loss for `G_AB`:** L1 norm between VGG features of `G_AB(X)` and `Y_target`. Weighted by `LAMBDA_CONTENT`.

    **Total Generator `G_AB` Loss (conceptual for its part):**
    `L_G_AB = L_adv_G_AB + LAMBDA_CYCLE * L_cycle_A + LAMBDA_L1 * L_L1_G_AB + LAMBDA_CONTENT * L_content_G_AB`

    **Total Generator `G_BA` Loss (conceptual for its part):**
    `L_G_BA = L_adv_G_BA + LAMBDA_CYCLE * L_cycle_B`

---

## 💾 Dataset

*   **Source:** [Sentinel12 Image Pairs Segregated by Terrain](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain) on Kaggle.
*   **Domains:**
    *   **Domain A (Input `X` for `G_AB`):** Sentinel-1 SAR grayscale images (`s1` folders).
    *   **Domain B (Target `Y` for `G_AB`):** Sentinel-2 Optical color images (`s2` folders).
*   **Path in Notebook:** The `DATA_DIR` is set to `/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2`.
*   **Preprocessing:** Images are resized (default 256x256), converted to tensors, and normalized to the `[-1, 1]` range.
*   **Subsetting:** The `SentinelDataset` class currently loads up to 1000 images per selected terrain class (`for s1_img_name in os.listdir(s1_path)[:1000]:`). The `terrain_classes_to_use` variable in the notebook is set to `['agri']` by default, meaning it will process up to 1000 images from the 'agri' class.

---

## 🛠️ Setup and Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Prerequisites:**
    *   Python 3.9+
    *   PyTorch (see `torch` version in notebook, typically 2.0+)
    *   Torchvision
    *   Pillow (PIL)
    *   NumPy
    *   Matplotlib
    *   tqdm
    *   Weights & Biases (`wandb`)
    *   `kaggle_secrets` (for running on Kaggle with W&B API key)

3.  **Install Dependencies (Recommended: use a virtual environment):**
    ```bash
    pip install torch torchvision torchaudio
    pip install Pillow numpy matplotlib tqdm wandb
    # If running locally and want to simulate Kaggle secrets, you might not need kaggle_secrets
    ```
    *(A `requirements.txt` file would be beneficial here).*

4.  **Weights & Biases API Key:**
    *   If running on Kaggle:
        *   Go to your Kaggle Notebook -> "Add-ons" -> "Secrets".
        *   Add a new secret with Label `WANDB_API_KEY` and your W&B API key as the Value.
    *   If running locally:
        *   Log in to W&B using the CLI: `wandb login`

---

## 🚀 Running the Notebook (`sarvision.ipynb`)

1.  **Environment:**
    *   **Recommended:** Kaggle Notebook with GPU enabled (P100 or T4).
    *   Can also be run locally if a CUDA-enabled GPU is available and dependencies are installed.

2.  **Dataset Path:**
    *   The `DATA_DIR` variable in the notebook is crucial.
        *   For Kaggle, it's set to: `/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2`
        *   Adjust if using a different path or running locally. The dummy data generation script might run if a local `./v2/` path is used and seems empty.

3.  **W&B Setup:**
    *   Ensure your W&B API key is configured (via Kaggle Secrets or local login).
    *   The `wandb.init()` call configures the project name (e.g., `SARcyclegan-colorization-advanced`). You can change this.

4.  **Hyperparameters:**
    *   Key hyperparameters like `LEARNING_RATE_GEN`, `BATCH_SIZE`, `NUM_EPOCHS`, loss weights (`LAMBDA_CYCLE`, `LAMBDA_L1`, `LAMBDA_CONTENT`), `IMG_SIZE` are defined at the top of the notebook and can be adjusted.

5.  **Execution:**
    *   Open `sarvision.ipynb` in Jupyter Lab/Notebook or Kaggle.
    *   Run the cells sequentially.

---

## 📈 Experiment Tracking with Weights & Biases

This project heavily utilizes W&B for comprehensive experiment tracking:

*   **Initialization:** `wandb.init()` logs all hyperparameters defined in its `config` argument.
*   **Metrics Logging:**
    *   **Batch-level:** Detailed losses for discriminators and generators (adversarial, cycle, L1, content) are logged for each batch.
    *   **Epoch-level:** Average generator and discriminator losses are logged at the end of each epoch.
*   **Image Logging:**
    *   Periodically (e.g., every 5 epochs), sample input grayscale images, target color images, and generated colorized images are logged to W&B under the "Media" tab.
*   **Model Artifacts:**
    *   Model checkpoints (`gen_AB.pth`, `gen_BA.pth`, etc.) are saved locally and also logged as W&B Artifacts. This allows for versioning and easy retrieval of trained models.
*   **System Metrics:** W&B automatically tracks GPU/CPU utilization, memory, etc.
*   **Dashboard:** All logged data can be visualized and compared across different runs on your W&B project dashboard.

---

## 🖼️ Outputs

*   **Locally Saved Images:** Sample generated images are saved to `/kaggle/working/output_images/` (or `./output_images/` if run locally and `OUTPUT_DIR` is adjusted).
*   **Locally Saved Checkpoints:** Model weights are saved to `/kaggle/working/checkpoints/`.
*   **W&B Artifacts:** Model checkpoints are also versioned and stored on W&B.
*   **W&B Media:** Generated image samples are viewable directly in the W&B run page.
*   **Loss Plots:** A plot of average G and D losses per epoch is saved locally and logged to W&B.

---

## 💡 Potential Improvements & Future Work

*   **Advanced Architectures:** Explore attention mechanisms (e.g., Self-Attention GAN) or more recent generator/discriminator designs.
*   **Loss Function Exploration:** Experiment with different GAN loss variants (e.g., Wasserstein GAN, Least Squares GAN) or alternative perceptual losses.
*   **Hyperparameter Optimization:** Utilize W&B Sweeps for systematic hyperparameter tuning.
*   **Larger Dataset / More Terrains:** Train on the full dataset or include more diverse terrain types for better generalization.
*   **Transfer Learning:** Investigate if pre-training on a larger, more general image-to-image translation dataset could improve performance.
*   **Quantitative Evaluation:** Implement quantitative metrics like PSNR, SSIM, LPIPS, FID for a more objective assessment of colorization quality.
*   **Deployment:** Package the trained `G_AB` generator for inference on new SAR images.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](<your-repo-url>/issues).

---

## 🙏 Acknowledgements

*   The original authors of CycleGAN, Pix2Pix, and the VGG network.
*   The creators of the Sentinel-1/2 dataset on Kaggle.
*   The PyTorch and Weights & Biases teams for their excellent tools.
