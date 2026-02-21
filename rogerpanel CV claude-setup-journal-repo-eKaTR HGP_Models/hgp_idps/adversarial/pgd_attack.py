"""
Projected Gradient Descent (PGD) Attack  (Appendix E, H.5)
==========================================================

PGD-10 with step size α = ε / K_PGD,  ε = 0.01 (L2, normalised features).
Euclidean ball projection with domain-specific validity constraints.
"""

import torch
import torch.nn as nn
from typing import Optional


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.01,
    steps: int = 10,
    step_size: Optional[float] = None,
    norm: str = "l2",
    clamp_min: float = -5.0,
    clamp_max: float = 5.0,
) -> torch.Tensor:
    """Generate adversarial examples via PGD.

    Parameters
    ----------
    model : nn.Module
        Model whose loss we maximise.
    x : Tensor [N, D]
        Clean inputs (normalised features).
    y : Tensor [N]
        Ground-truth binary labels.
    epsilon : float
        Maximum perturbation budget.
    steps : int
        Number of PGD iterations (K_PGD = 10 in paper).
    step_size : float or None
        Per-step size; defaults to epsilon / steps.
    norm : str
        "l2" or "linf".
    clamp_min, clamp_max : float
        Feature validity bounds (post-standardisation).

    Returns
    -------
    x_adv : Tensor [N, D]
    """
    if step_size is None:
        step_size = epsilon / steps

    x_adv = x.clone().detach()
    x_adv += torch.empty_like(x_adv).uniform_(-epsilon * 0.1, epsilon * 0.1)
    x_adv = torch.clamp(x_adv, clamp_min, clamp_max)

    for _ in range(steps):
        x_adv.requires_grad_(True)

        # Surrogate loss — binary cross-entropy on mean prediction
        if hasattr(model, "forward"):
            out = model(x_adv)
            if hasattr(out, "mean"):
                pred = out.mean
            else:
                pred = out
        else:
            pred = x_adv.mean(dim=1)

        loss = nn.functional.binary_cross_entropy_with_logits(
            pred.view(-1), y.float().view(-1)
        )
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
        x_adv = x_adv.detach()

        if norm == "l2":
            grad_norm = grad.norm(dim=1, keepdim=True).clamp(min=1e-8)
            x_adv = x_adv + step_size * grad / grad_norm
            # Project back onto L2 ball
            delta = x_adv - x
            delta_norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-8)
            factor = torch.min(
                torch.ones_like(delta_norm), epsilon / delta_norm
            )
            x_adv = x + delta * factor
        else:  # linf
            x_adv = x_adv + step_size * grad.sign()
            delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = x + delta

        x_adv = torch.clamp(x_adv, clamp_min, clamp_max)

    return x_adv.detach()
