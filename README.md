# PGMP: Physically-Grounded Manifold Projection for Dental CBCT MAR

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2025.xxxxx) 
[![Framework](https://img.shields.io/badge/PyTorch-1.13+-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![SOTA](https://img.shields.io/badge/SOTA-Yes-brightgreen.svg)]()

> **Physically-Grounded Manifold Projection Model for Generalizable Metal Artifact Reduction in Dental CBCT**  
> *Zhi Li\*, Yaqi Wang\*, Bingtao Ma, Yifan Zhang, Huiyu Zhou, Shuai Wang*  
> (\* Equal Contribution)  
---

## 📖 Introduction

**PGMP** is a novel framework for Metal Artifact Reduction (MAR) in Dental CBCT. It addresses the "synthetic-to-real" domain gap and the inefficiency of stochastic diffusion models. 

By shifting the paradigm from iterative noise prediction ($\epsilon$-prediction) to **Direct Manifold Projection ($x$-prediction)**, and leveraging **Medical Foundation Models (MedDINOv3)** for semantic guidance, PGMP achieves State-of-the-Art (SOTA) restoration quality with real-time inference speed.

<p align="center">
  <img src="assets/teaser.png" width="800" title="Framework Overview">
</p>

### ✨ Key Features

- **⚛️ Anatomically-Adaptive Physics Simulation (AAPS):**  
  Bridges the domain gap using patient-specific "Digital Twins" and Monte Carlo-based spectral hardening modeling. No more simplistic ray-tracing.
- **🚀 DMP-Former (Direct Manifold Projection):**  
  A deterministic transformer that projects corrupted data directly onto the clean anatomical manifold in a **single step**.  
  *Speedup:* **1000×** faster than standard DDPM, **50×** faster than DDIM.
- **🧠 Semantic-Structural Alignment (SSA):**  
  Distills expert-level priors from **MedDINOv3** to prevent structural hallucinations and preserve diagnostic details (e.g., trabecular bone).

---

## 🏆 Performance

PGMP outperforms current SOTA methods (supervised, semi-supervised, and diffusion-based) on both synthetic and clinical datasets.

| Method | Backbone | PSNR (dB) ⬆ | SSIM ⬆ | Inference Time ⬇ |
| :--- | :---: | :---: | :---: | :---: |
| CNNMAR | CNN | 30.59 | 0.8676 | Fast |
| DuDoNet++ | Dual-Domain | 36.75 | 0.9050 | Medium |
| DuDoDp | Diffusion | 32.61 | 0.8463 | Very Slow |
| **PGMP (Ours)** | **DMP-Former** | **36.80** | **0.9165** | **Fast** |

> *Evaluated on the AAPS large-scale test set.*

---

## 🛠️ Methodology

The framework consists of three synergistic components:

1.  **Data Engine (AAPS):** Generates high-fidelity training pairs with polychromatic X-ray simulation and clinically valid metal placement.
2.  **Network (DMP-Former):** An isotropic ViT with AdaLN-Zero and RoPE attention, optimized for direct manifold reconstruction.
3.  **Loss Function:** A composite objective combining Pixel Reconstruction ($L_1$), Semantic Alignment ($L_{SSA}$), and Structural Edge Consistency ($L_{edge}$).

$$ \mathcal{L}_{total} = \|\hat{x} - x_{gt}\|_1 + \lambda_{SSA} \mathcal{L}_{SSA} + \lambda_{edge} \|\nabla\hat{x} - \nabla x_{gt}\|_1 $$

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/yourusername/PGMP-MAR.git
cd PGMP-MAR
pip install -r requirements.txt
```
### 2. Data Preparation
Download the STS24 dataset or use your own CBCT volumes.
```bash
# Generate AAPS synthetic data (Physics Simulation)
python data/generate_aaps.py --source_path ./STS24 --output_path ./data/train
```
### 3. Training
Train the DMP-Former with MedDINOv3 guidance.
``` bash
python train.py --config configs/pgmp_train.yaml
```
### 4. Inference (Demo)
Run MAR on a clinical sample
```bash
python inference.py --checkpoint weights/pgmp_best.pth --input assets/clinical_sample.dcm
```
##  🖼️ Gallery
Clinical Generalization
Comparison on real-world clinical data (Zero-shot). Note the preservation of trabecular bone texture in our result.
<p align="center">
<img src="assets/results_clinical.png" width="800">
</p>
Downstream Task (Segmentation)
PGMP significantly improves the accuracy of automated tooth segmentation (Dice +5.4% in Maxilla).
<p align="center">
<img src="assets/segmentation.png" width="800">
</p>

## 📝 Citation

If you find this project useful for your research, please consider citing:

```bibtex
@article{Li2025PGMP,
  title={Physically-Grounded Manifold Projection Model for Generalizable Metal Artifact Reduction in Dental CBCT},
  author={Li, Zhi and Wang, Yaqi and Ma, Bingtao and Zhang, Yifan and Zhou, Huiyu and Wang, Shuai},
  journal={arXiv preprint arXiv:2025.xxxxx}, 
  year={2025}
}

## 📧 Contact
For questions or collaboration, please contact Shuai Wang at zhi.li@hdu.edu.cn .
