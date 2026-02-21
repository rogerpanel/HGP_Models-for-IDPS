"""
Uncertainty-Calibrated Hierarchical Gaussian Processes
for Intrusion Detection with Multi-Scale Temporal Modeling
==========================================================

Reference implementation accompanying the Neurocomputing submission:
  "Uncertainty-Calibrated Hierarchical Gaussian Processes for Intrusion
   Detection with Multi-Scale Temporal Modeling"

Modules
-------
models      – Hierarchical GP and Deep-Kernel projection network
kernels     – Multi-scale temporal, domain-specific, and centered kernels
adversarial – Adversarially-robust inducing-point optimisation (Algorithm 1)
detection   – Uncertainty-calibrated anomaly scoring & 3-tier alerting
data        – ICS3D loader and preprocessing pipeline
training    – ELBO training loop with cosine-annealing & online updates
evaluation  – Metrics, visualisation, ablation, and cross-domain transfer
"""

__version__ = "3.0.0"
