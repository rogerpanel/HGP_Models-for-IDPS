# Uncertainty-Calibrated Hierarchical Gaussian Processes for Intrusion Detection with Multi-Scale Temporal Modeling

Reference implementation for the Neurocomputing submission.

## Overview

This repository provides the **complete, reproducible implementation** of the Hierarchical Gaussian Process (HGP) framework for uncertainty-aware intrusion detection across heterogeneous cloud security domains. The system processes data from three distinct security environments — Edge-IIoT, containerised microservices, and Security Operations Centres — through a unified probabilistic architecture that provides calibrated confidence estimates alongside every detection decision.

### Key Capabilities

| Capability | Description | Paper Reference |
|---|---|---|
| **Deep-Kernel Projection** | Neural feature extractor π : X^(d) → Z mapping heterogeneous inputs to a shared latent space | Section IV.A, Eq. 25–27 |
| **Hierarchical GP Decomposition** | Additive decomposition into shared, domain-specific, temporal, and interaction components | Proposition 1 |
| **Multi-Scale Temporal Kernels** | Seven-component RBF bank (μs → weeks), periodic, spectral mixture, and change-point kernels | Section IV.B, Eq. 33–35 |
| **Kernel Centering** | Functional ANOVA centering for identifiable variance attribution | Appendix D.4 |
| **Adversarial Inducing Points** | Class-weighted, PGD-robustified inducing-point selection | Algorithm 1, Appendix E |
| **Uncertainty-Calibrated Detection** | Epistemic/aleatoric decomposition with domain-adaptive thresholds | Section IV.D, Eq. 39–40 |
| **3-Tier Alert Prioritisation** | σ-based escalation: log / queue / page | Section IV.D.3 |
| **Incremental Posterior Updates** | Streaming-compatible online variational updates | Appendix G.1 |
| **Cross-Domain Transfer** | Fine-tune from data-rich to data-scarce domains | Section VII.E |

## Dataset

**Integrated Cloud Security 3-Datasets (ICS3D)** — 21.48 M records

| Domain | Records | Features | Attack Rate |
|--------|---------|----------|-------------|
| Edge-IIoT (DNN) | 2,219,000 | 60–140 | 27.3% |
| Container | 234,560 | 87 | 6.1% |
| SOC (GUIDE) | 16,950,000 | 33 entity types | 0.8% |

DOI: [10.34740/kaggle/dsv/12483891](https://doi.org/10.34740/kaggle/dsv/12483891)

The dataset is automatically downloaded via `kagglehub` when running the notebook.

## Repository Structure

```
HGP_Models/
├── README.md
├── requirements.txt
├── configs/
│   └── default_config.yaml          # Full configuration (matches Appendix H)
├── hgp_idps/                        # Python package
│   ├── models/
│   │   ├── projection_network.py    # Deep-kernel π(·)  [128, 64, 32]
│   │   ├── hierarchical_gp.py       # Sparse variational GP
│   │   └── deep_kernel_gp.py        # End-to-end model (π + GP + likelihood)
│   ├── kernels/
│   │   ├── multiscale_temporal.py   # RBF bank + periodic + spectral + changepoint
│   │   ├── domain_specific.py       # Edge-IIoT / Container / SOC kernels
│   │   └── centered_kernel.py       # Functional ANOVA centering
│   ├── adversarial/
│   │   ├── pgd_attack.py            # PGD-10 (L2, ε = 0.01)
│   │   └── inducing_points.py       # Algorithm 1: robust inducing-point optimisation
│   ├── detection/
│   │   ├── uncertainty.py           # Uncertainty decomposition + anomaly scoring
│   │   └── alert_prioritization.py  # 3-tier alerting
│   ├── data/
│   │   ├── loader.py                # ICS3D download & load
│   │   └── preprocessing.py         # Unified 5-step preprocessing pipeline
│   ├── training/
│   │   ├── trainer.py               # Full training loop (Algorithm 2)
│   │   └── online_update.py         # Incremental posterior updates
│   └── evaluation/
│       ├── metrics.py               # Accuracy, F1, AUC-ROC, ECE, per-attack F1
│       ├── visualization.py         # Publication-quality figures (3×3 grid)
│       └── ablation.py              # Component ablation + cross-domain transfer
├── notebooks/
│   └── hgp_model_v3.ipynb           # Main executable notebook
└── results/                         # Generated figures and reports
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

Open `notebooks/hgp_model_v3.ipynb` in Jupyter or on Kaggle and execute all cells. The notebook:

1. Downloads the ICS3D dataset from Kaggle
2. Preprocesses all three domains
3. Builds the hierarchical kernel with centering
4. Initialises adversarially-robust inducing points
5. Trains with cosine-annealing, gradient accumulation, and early stopping
6. Evaluates detection performance, calibration, and adversarial robustness
7. Runs ablation studies and cross-domain transfer
8. Generates all visualisations

### 3. Use as a library

```python
from hgp_idps.models import DeepKernelHGP
from hgp_idps.detection import UncertaintyCalibratedDetector, AlertPrioritizer
from hgp_idps.training import HierarchicalGPTrainer

# Build and train
trainer = HierarchicalGPTrainer(config=train_cfg, device=device)
model = trainer.build_model(input_dim=87, kernel=kernel, X_train=X, y_train=y)
trainer.train(train_loader, val_loader)

# Detect with uncertainty
detector = UncertaintyCalibratedDetector(model)
results = detector.detect(X_new, domain='container')

# Prioritise alerts
prioritizer = AlertPrioritizer()
alerts = prioritizer.prioritize(results)
```

## Training Configuration

Key hyperparameters (full specification in `configs/default_config.yaml`):

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Projection layers | [128, 64, 32] | Section IV.A |
| Activation | SiLU | Section IV.A |
| Inducing points | 500 (Edge), 300 (Container), 200 (SOC) | Appendix H.3 |
| PGD steps | 10, ε = 0.01 | Appendix H.5 |
| Epochs | 100 (early stopping patience = 10) | Appendix H.3 |
| Batch size | 2048 (Edge), 1024 (Container), 4096 (SOC) | Appendix H.3 |
| Optimiser | Adam (η = 10⁻³, β₁ = 0.9, β₂ = 0.999) | Appendix H.3 |
| Scheduler | Cosine annealing, T₀ = 20, T_mult = 2 | Appendix H.3 |
| Gradient clipping | max norm = 1.0 | Appendix H.3 |
| Gradient accumulation | 4 steps | Appendix H.3 |

## Software Stack

- PyTorch ≥ 2.1.0 with CUDA 12.1
- GPyTorch ≥ 1.11
- NumPy ≥ 1.24.3, SciPy ≥ 1.11.1
- Scikit-learn ≥ 1.3.0

## Improvements over v2 (Initial Submission)

| Aspect | v2 (Initial) | v3 (Revised) |
|--------|-------------|-------------|
| Feature extraction | Direct features | Deep-kernel projection π(·) with [128, 64, 32] SiLU network |
| Kernel identifiability | None | Functional ANOVA centering (Appendix D.4) |
| Inducing-point allocation | Uniform k-means | Class-weighted √(1/n_c) allocation (Eq. 36) |
| Temporal modelling | 7 RBF + 3 periodic | + spectral mixture + sigmoid change-point kernels |
| Training schedule | Fixed LR, 50 epochs | Cosine warm restarts, gradient accumulation, 100 epochs |
| Adversarial training | Basic PGD | PGD-10 with L2 projection, every 5 epochs |
| Alert system | Binary detection | 3-tier prioritisation (log / queue / page) |
| Online learning | None | Incremental posterior updates (Appendix G.1) |
| Evaluation | Basic metrics | + ECE calibration, per-attack F1, ablation, cross-domain transfer |
| Code structure | Single notebook | Modular Python package + notebook |

## Citation

If you use this code, please cite:

```bibtex
@article{hgp_idps_2025,
  title={Uncertainty-Calibrated Hierarchical Gaussian Processes for Intrusion
         Detection with Multi-Scale Temporal Modeling},
  journal={Neurocomputing},
  year={2025}
}
```

## License

MIT
