"""
Adversarially-Robust Inducing Point Optimisation  (Algorithm 1, Appendix E)
===========================================================================

Nested optimisation loop:
  Outer — update inducing locations Z via Adam  (η = 1e-3)
  Inner — PGD-10 attack to generate worst-case perturbations

Class-weighted allocation:  M_c^(d) = M_d · √(1/n_c) / Σ √(1/n_c')
"""

import torch
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional

from .pgd_attack import pgd_attack


class AdversarialInducingPointOptimizer:
    """Select and robustify inducing-point locations.

    Parameters
    ----------
    epsilon : float
        PGD perturbation budget.
    pgd_steps : int
        Inner PGD iterations.
    outer_iterations : int
        Outer optimisation steps for Z.
    lr : float
        Adam learning rate for Z.
    class_weighted : bool
        Use √(1/n_c) weighting for class-balanced allocation.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        pgd_steps: int = 10,
        outer_iterations: int = 20,
        lr: float = 1e-3,
        class_weighted: bool = True,
    ):
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.outer_iterations = outer_iterations
        self.lr = lr
        self.class_weighted = class_weighted

    def select_inducing_points(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_inducing: int = 500,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Algorithm 1: adversarially-robust inducing-point optimisation.

        Steps
        -----
        1.  Class-weighted k-means initialisation.
        2.  Outer loop: PGD worst-case → coverage + diversity loss → Adam update.

        Returns
        -------
        Z : Tensor [M, D]  — robustified inducing locations.
        """
        device = device or X.device
        num_inducing = min(num_inducing, X.shape[0] // 5)
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        # --- Step 1: class-weighted k-means ---
        if self.class_weighted:
            Z_init = self._class_weighted_kmeans(X_np, y_np, num_inducing)
        else:
            km = KMeans(n_clusters=num_inducing, random_state=42, n_init=10)
            km.fit(X_np)
            Z_init = km.cluster_centers_

        Z = torch.tensor(Z_init, dtype=torch.float32, device=device)
        Z.requires_grad_(True)
        optimizer = optim.Adam([Z], lr=self.lr)

        # Subsample for efficiency
        max_sub = min(2048, X.shape[0])
        idx = torch.randperm(X.shape[0])[:max_sub]
        X_sub = X[idx].to(device)
        y_sub = y[idx].to(device)

        print(f"  [Adversarial inducing-point optimisation]  M={num_inducing}")

        # --- Step 2: outer optimisation loop ---
        for it in range(self.outer_iterations):
            # Inner: PGD perturbation of data
            X_adv = self._simple_pgd(X_sub, y_sub)

            # Coverage loss: mean min-distance from perturbed data to Z
            dists = torch.cdist(X_adv, Z)                    # [N, M]
            coverage_loss = dists.min(dim=1)[0].mean()

            # Diversity loss: discourage Z from collapsing
            Z_dists = torch.cdist(Z, Z) + torch.eye(
                Z.size(0), device=device
            ) * 1e6
            diversity_loss = -torch.log(Z_dists.min(dim=1)[0] + 1e-8).mean()

            loss = coverage_loss + 0.1 * diversity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (it + 1) % 10 == 0 or it == 0:
                print(
                    f"    iter {it+1:3d}/{self.outer_iterations}  "
                    f"loss={loss.item():.4f}  "
                    f"coverage={coverage_loss.item():.4f}  "
                    f"diversity={diversity_loss.item():.4f}"
                )

        return Z.detach()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _class_weighted_kmeans(
        self, X: np.ndarray, y: np.ndarray, M: int
    ) -> np.ndarray:
        """Allocate M_c = M · √(1/n_c) / Σ √(1/n_c') inducing points per class."""
        classes, counts = np.unique(y, return_counts=True)
        weights = np.sqrt(1.0 / counts)
        weights /= weights.sum()
        allocations = np.maximum((weights * M).astype(int), 1)

        # Adjust to exactly M
        diff = M - allocations.sum()
        if diff > 0:
            allocations[np.argmax(weights)] += diff
        elif diff < 0:
            allocations[np.argmax(allocations)] += diff

        centers = []
        for cls, n_cls in zip(classes, allocations):
            X_cls = X[y == cls]
            n_cls = min(n_cls, X_cls.shape[0])
            if n_cls <= 0:
                continue
            km = KMeans(n_clusters=n_cls, random_state=42, n_init=5)
            km.fit(X_cls)
            centers.append(km.cluster_centers_)

        return np.vstack(centers)

    def _simple_pgd(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Lightweight PGD for inducing-point robustification."""
        X_adv = X.clone().detach()
        X_adv.requires_grad_(True)

        for _ in range(self.pgd_steps):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                X_adv.mean(dim=1), y.float()
            )
            grad = torch.autograd.grad(loss, X_adv)[0]
            X_adv = X_adv.detach() + (self.epsilon / self.pgd_steps) * grad.sign()
            delta = torch.clamp(X_adv - X, -self.epsilon, self.epsilon)
            X_adv = (X + delta).detach().requires_grad_(True)

        return X_adv.detach()
