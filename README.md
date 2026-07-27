# Physics-informed self-supervised denoising for fluorescence microscopy via structure and intensity priors (SIN2V)

## ⚠️ Code Availability & Patent Notice
The core source code of **SIN2V** is provided as Supplementary Software for the purpose of editorial and peer review. Due to pending patent applications, the public repository on GitHub is currently restricted. A fully documented, open-source version of the code will be made publicly available on GitHub and archived with a permanent DOI upon formal publication of the manuscript.

---

## 💡 Overview
This repository contains the official PyTorch implementation of **SIN2V**, a physics-informed, fully self-supervised blind denoising algorithm designed specifically for fluorescence microscopy. It requires **NO ground truth (clean) images** for training. 

Fluorescence images typically suffer from extremely low Signal-to-Noise Ratios (SNR) and highly uneven intensity distributions. To address this, SIN2V integrates two key physics-informed priors into the standard Noise2Void framework:
1. **Structure Prior (Variance-based Probabilistic Masking):** Adaptively senses local structural complexity to protect fragile biological structures (e.g., cytoskeletons) from being over-smoothed.
2. **Intensity Prior (Luminance-Weighted Loss):** Balances the extreme gradient variations between bright fluorescent spots and dim background structures.

---

## 📂 Repository Structure
Please ensure your unzipped directory is organized as follows before running the scripts:

```text
SIN2V_Supplementary_Software/
│
├── train.py               # Main script to train the SIN2V model from scratch
├── trainer.py             # Training loop, loss computation, and validation logic
├── dataset.py             # Dynamic dataloader with background filtering
├── utils.py               # Core physics priors (Probabilistic Mask & Luminance Loss)
├── model.py               # Network architecture definition (PolyMambaUNet)
├── config.yaml            # Hyperparameter configuration file
├── requirements.txt       # Python environment dependencies
