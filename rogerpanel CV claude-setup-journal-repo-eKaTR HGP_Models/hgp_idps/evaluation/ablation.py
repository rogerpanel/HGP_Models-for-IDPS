"""
Ablation Studies & Cross-Domain Transfer  (Section VII.D-E, Appendix L.3)
=========================================================================

1.  Component ablation: measure F1 drop when removing each kernel component.
2.  Cross-domain transfer: train on source domain, fine-tune on target with
    limited labelled data.
"""

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from ..training.trainer import HierarchicalGPTrainer


# ------------------------------------------------------------------ #
#  Ablation study                                                      #
# ------------------------------------------------------------------ #
def run_ablation_study(
    base_trainer: HierarchicalGPTrainer,
    test_loader: DataLoader,
    component_names: List[str],
    domain: str = "multi",
) -> Dict[str, Dict[str, float]]:
    """Ablation: for each kernel component, zero its scale and measure F1 drop.

    Parameters
    ----------
    base_trainer : HierarchicalGPTrainer
        Fully-trained trainer with model.
    test_loader : DataLoader
    component_names : list[str]
        Names of kernel components to ablate.
    domain : str

    Returns
    -------
    dict  {component_name: {"f1": ..., "f1_drop": ...}}
    """
    model = base_trainer.model
    device = base_trainer.device

    # Full-model baseline
    baseline_f1 = _eval_f1(model, test_loader, device)
    print(f"\n  Ablation baseline F1: {baseline_f1:.4f}")

    results = {"full_model": {"f1": baseline_f1, "f1_drop": 0.0}}

    kernel = model.gp.covar_module  # CenteredAdditiveKernel
    original_scales = kernel.log_scales.data.clone()

    for idx, name in enumerate(kernel.component_names):
        if name not in component_names:
            continue

        # Zero this component's scale
        kernel.log_scales.data[idx] = -100.0  # softplus(-100) ≈ 0

        ablated_f1 = _eval_f1(model, test_loader, device)
        drop = baseline_f1 - ablated_f1
        results[name] = {"f1": ablated_f1, "f1_drop": drop}
        print(f"    -{name:30s}  F1={ablated_f1:.4f}  drop={drop:+.4f}")

        # Restore
        kernel.log_scales.data.copy_(original_scales)

    return results


# ------------------------------------------------------------------ #
#  Cross-domain transfer                                               #
# ------------------------------------------------------------------ #
def run_cross_domain_transfer(
    source_trainer: HierarchicalGPTrainer,
    target_X: torch.Tensor,
    target_y: torch.Tensor,
    target_domain: str,
    fine_tune_epochs: int = 20,
    fine_tune_fraction: float = 0.1,
) -> Dict[str, float]:
    """Transfer from source domain → target domain with limited labels.

    1.  Copy source model.
    2.  Fine-tune on `fine_tune_fraction` of target labels.
    3.  Evaluate on remaining target data.

    Returns dict with accuracy, f1 for both zero-shot and fine-tuned.
    """
    device = source_trainer.device
    model = copy.deepcopy(source_trainer.model).to(device)

    n = target_X.size(0)
    n_ft = max(int(n * fine_tune_fraction), 64)
    perm = torch.randperm(n)
    ft_idx, eval_idx = perm[:n_ft], perm[n_ft:]

    X_ft = target_X[ft_idx].to(device)
    y_ft = target_y[ft_idx].to(device)
    X_eval = target_X[eval_idx].to(device)
    y_eval = target_y[eval_idx].cpu().numpy()

    # Zero-shot evaluation
    zs_f1 = _eval_f1_tensor(model, X_eval, y_eval, device)
    print(f"  Zero-shot F1 on {target_domain}: {zs_f1:.4f}")

    # Fine-tune
    model.train()
    model.gp.train()
    model.likelihood.train()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    from gpytorch.mlls import VariationalELBO
    mll = VariationalELBO(model.likelihood, model.gp, num_data=n_ft)

    ft_loader = DataLoader(TensorDataset(X_ft, y_ft), batch_size=256, shuffle=True)
    for ep in range(fine_tune_epochs):
        for bx, by in ft_loader:
            opt.zero_grad()
            out = model(bx)
            loss = -mll(out, by.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    ft_f1 = _eval_f1_tensor(model, X_eval, y_eval, device)
    improvement = ft_f1 - zs_f1
    print(f"  Fine-tuned F1 on {target_domain}: {ft_f1:.4f}  (Δ={improvement:+.4f})")

    return {
        "zero_shot_f1": zs_f1,
        "fine_tuned_f1": ft_f1,
        "improvement": improvement,
    }


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #
def _eval_f1(model, loader, device) -> float:
    model.eval()
    model.gp.eval()
    model.likelihood.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            out = model(bx)
            pred = model.likelihood(out).mean.round().cpu().numpy()
            all_pred.append(pred)
            all_true.append(by.numpy())
    return f1_score(np.concatenate(all_true), np.concatenate(all_pred), zero_division=0)


def _eval_f1_tensor(model, X, y_np, device) -> float:
    model.eval()
    model.gp.eval()
    model.likelihood.eval()
    preds = []
    bs = 512
    with torch.no_grad():
        for i in range(0, X.size(0), bs):
            bx = X[i: i + bs].to(device)
            out = model(bx)
            pred = model.likelihood(out).mean.round().cpu().numpy()
            preds.append(pred)
    return f1_score(y_np[:len(np.concatenate(preds))], np.concatenate(preds), zero_division=0)
