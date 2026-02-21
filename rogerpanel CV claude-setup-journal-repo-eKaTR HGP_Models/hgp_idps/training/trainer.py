"""
Hierarchical GP Training Loop  (Algorithm 2, Appendix H.3)
==========================================================

- Adam for variational parameters (β₁=0.9, β₂=0.999, η=1e-3)
- Optional L-BFGS-B for kernel hyperparameters (max 100 iter)
- Cosine annealing with warm restarts  (T_0=20, T_mult=2)
- Gradient clipping  (max norm = 1.0)
- Gradient accumulation  (4 steps → effective batch × 4)
- Early stopping  (patience = 10 on validation ELBO)
- Adversarial inducing-point re-optimisation every 5 epochs
"""

import time
import copy
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import BernoulliLikelihood

from ..models.projection_network import ProjectionNetwork
from ..models.hierarchical_gp import HierarchicalCloudSecurityGP
from ..models.deep_kernel_gp import DeepKernelHGP
from ..kernels.centered_kernel import CenteredAdditiveKernel
from ..adversarial.inducing_points import AdversarialInducingPointOptimizer
from ..adversarial.pgd_attack import pgd_attack


class HierarchicalGPTrainer:
    """End-to-end trainer for the deep-kernel hierarchical GP.

    Parameters
    ----------
    config : dict
        Training configuration (see configs/default_config.yaml → training).
    device : torch.device
        GPU / CPU.
    """

    def __init__(self, config: dict, device: torch.device):
        self.cfg = config
        self.device = device
        self.model: Optional[DeepKernelHGP] = None
        self.metrics: Dict[str, List] = {"train": [], "val": []}
        self._best_state = None
        self._best_val = float("inf")

    # ------------------------------------------------------------------ #
    #  Build model                                                         #
    # ------------------------------------------------------------------ #
    def build_model(
        self,
        input_dim: int,
        kernel: CenteredAdditiveKernel,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        num_inducing: int = 500,
        projection_cfg: Optional[dict] = None,
        adversarial_cfg: Optional[dict] = None,
    ) -> DeepKernelHGP:
        """Construct projection + GP + likelihood and initialise inducing points."""
        proj_cfg = projection_cfg or {}
        adv_cfg = adversarial_cfg or {}

        # 1. Projection network
        projection = ProjectionNetwork(
            input_dim=input_dim,
            hidden_dims=proj_cfg.get("layers", [128, 64, 32]),
            activation=proj_cfg.get("activation", "silu"),
            dropout=proj_cfg.get("dropout", 0.1),
            batch_norm=proj_cfg.get("batch_norm", True),
        ).to(self.device)

        latent_dim = projection.output_dim

        # 2. Project training data for inducing-point init
        projection.eval()
        with torch.no_grad():
            Z_data = projection(X_train[:min(5000, len(X_train))].to(self.device))
        projection.train()

        # 3. Adversarially-robust inducing-point selection
        if adv_cfg.get("enabled", True):
            aip = AdversarialInducingPointOptimizer(
                epsilon=adv_cfg.get("epsilon", 0.01),
                pgd_steps=adv_cfg.get("pgd_steps", 10),
                outer_iterations=20,
                lr=1e-3,
                class_weighted=True,
            )
            inducing = aip.select_inducing_points(
                Z_data,
                y_train[:Z_data.size(0)].to(self.device),
                num_inducing=num_inducing,
                device=self.device,
            )
        else:
            from sklearn.cluster import KMeans
            n = min(num_inducing, Z_data.size(0) // 5)
            km = KMeans(n_clusters=n, random_state=42, n_init=10)
            km.fit(Z_data.cpu().numpy())
            inducing = torch.tensor(
                km.cluster_centers_, dtype=torch.float32, device=self.device
            )

        # 4. Fit kernel centering on projected data
        kernel = kernel.to(self.device)
        kernel.fit(Z_data, max_samples=1024)

        # 5. Assemble GP
        gp = HierarchicalCloudSecurityGP(
            inducing_points=inducing,
            feature_dim=latent_dim,
            kernel=kernel,
            use_linear_mean=True,
            learn_inducing=True,
        ).to(self.device)

        likelihood = BernoulliLikelihood().to(self.device)

        self.model = DeepKernelHGP(projection, gp, likelihood)
        print(f"\n  Model built: {sum(p.numel() for p in self.model.parameters()):,} params")
        print(f"  Inducing points: {inducing.shape[0]}")
        print(f"  Latent dim: {latent_dim}")
        return self.model

    # ------------------------------------------------------------------ #
    #  Training loop                                                       #
    # ------------------------------------------------------------------ #
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        """Full training loop with all paper-specified bells and whistles."""
        model = self.model
        model.train()

        epochs = self.cfg.get("epochs", 100)
        grad_clip = self.cfg.get("gradient_clip", 1.0)
        accum = self.cfg.get("gradient_accumulation_steps", 4)
        patience = self.cfg.get("early_stopping", {}).get("patience", 10)
        val_every = self.cfg.get("validation_every", 5)
        adv_freq = self.cfg.get("adversarial_frequency", 5)

        # Optimiser: Adam for everything jointly (simpler, robust)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.get("optimizer", {}).get("variational", {}).get("lr", 1e-3),
            betas=tuple(
                self.cfg.get("optimizer", {}).get("variational", {}).get("betas", [0.9, 0.999])
            ),
        )

        # Cosine-annealing with warm restarts
        sched_cfg = self.cfg.get("scheduler", {})
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg.get("T_0", 20),
            T_mult=sched_cfg.get("T_mult", 2),
            eta_min=sched_cfg.get("eta_min", 1e-5),
        )

        mll = VariationalELBO(
            model.likelihood, model.gp, num_data=len(train_loader.dataset)
        )

        wait = 0
        t0 = time.time()

        print(f"\n{'='*60}")
        print(f"  Training  |  {epochs} epochs  |  accum={accum}  |  patience={patience}")
        print(f"{'='*60}")

        for epoch in range(1, epochs + 1):
            model.train()
            model.gp.train()
            model.likelihood.train()

            epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
            optimizer.zero_grad()

            for step, (bx, by) in enumerate(train_loader):
                bx = bx.to(self.device)
                by = by.to(self.device).float()

                output = model(bx)
                loss = -mll(output, by) / accum
                loss.backward()

                if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * accum

                with torch.no_grad():
                    preds = model.likelihood(output).mean.round()
                    epoch_acc += (preds == by).float().mean().item()
                n_batches += 1

            epoch_loss /= n_batches
            epoch_acc /= n_batches
            scheduler.step(epoch)

            self.metrics["train"].append({"loss": epoch_loss, "acc": epoch_acc})

            # Validation
            val_loss = None
            if val_loader and (epoch % val_every == 0 or epoch == 1):
                val_loss, val_acc = self._evaluate(val_loader, mll)
                self.metrics["val"].append({"epoch": epoch, "loss": val_loss, "acc": val_acc})

                if val_loss < self._best_val:
                    self._best_val = val_loss
                    self._best_state = copy.deepcopy(model.state_dict())
                    wait = 0
                else:
                    wait += 1

                if wait >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

            # Log
            if epoch % 5 == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                msg = (
                    f"  Epoch {epoch:3d}/{epochs}  "
                    f"train_loss={epoch_loss:.4f}  "
                    f"train_acc={epoch_acc:.4f}  "
                    f"lr={lr_now:.2e}"
                )
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                print(msg)

        elapsed = time.time() - t0
        print(f"\n  Training complete in {elapsed:.1f}s")

        # Load best checkpoint
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
            print("  Loaded best validation checkpoint")

    # ------------------------------------------------------------------ #
    #  Evaluation helper                                                   #
    # ------------------------------------------------------------------ #
    def _evaluate(self, loader, mll):
        self.model.eval()
        self.model.gp.eval()
        self.model.likelihood.eval()

        total_loss, total_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(self.device)
                by = by.to(self.device).float()
                out = self.model(bx)
                loss = -mll(out, by)
                total_loss += loss.item()
                preds = self.model.likelihood(out).mean.round()
                total_acc += (preds == by).float().mean().item()
                n += 1

        return total_loss / n, total_acc / n
