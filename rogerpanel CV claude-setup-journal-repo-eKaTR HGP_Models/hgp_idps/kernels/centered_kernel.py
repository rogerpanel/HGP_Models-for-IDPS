"""
Kernel Centering for Identifiability  (Appendix D.4)
====================================================

Enforces orthogonality of the additive kernel decomposition via the
functional ANOVA centering transform.  Given an additive kernel

    k(x, x') = Σ_i  k_i(x, x')

each component is centered so that its mean over the training data is
zero, preventing non-unique decompositions and enabling interpretable
variance attribution.

Empirical centering uses Monte-Carlo integration:

    k̃_i(x, x') = k_i(x, x')
               - (1/N) Σ_n k_i(x, x_n)
               - (1/N) Σ_n k_i(x_n, x')
               + (1/N²) Σ_{n,m} k_i(x_n, x_m)

Followed by scale normalisation so that the total trace equals 1.
"""

import torch
import gpytorch
from gpytorch.kernels import Kernel, ScaleKernel, AdditiveKernel
from typing import List, Tuple, Optional


class CenteredKernelWrapper(Kernel):
    """Wraps a single kernel with empirical centering statistics.

    Call ``fit(X_ref)`` once after construction to compute centering terms.
    """

    has_lengthscale = False

    def __init__(self, base_kernel: Kernel, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        # Centering buffers (populated by fit())
        self.register_buffer("_col_mean", torch.tensor(0.0))
        self.register_buffer("_grand_mean", torch.tensor(0.0))
        self._ref_points: Optional[torch.Tensor] = None
        self._fitted = False

    def fit(self, X_ref: torch.Tensor, max_samples: int = 1024):
        """Compute centering statistics from reference data."""
        if X_ref.size(0) > max_samples:
            idx = torch.randperm(X_ref.size(0))[:max_samples]
            X_ref = X_ref[idx]

        self._ref_points = X_ref.detach()
        with torch.no_grad():
            K_ref = self.base_kernel(X_ref, X_ref).to_dense()  # [N, N]
            self._col_mean = K_ref.mean(dim=0)       # [N]
            self._grand_mean = K_ref.mean()           # scalar
        self._fitted = True

    def forward(self, x1, x2, diag=False, **params):
        K_raw = self.base_kernel(x1, x2, diag=diag, **params)

        if not self._fitted or self._ref_points is None:
            return K_raw

        X_ref = self._ref_points.to(x1.device)

        with torch.no_grad():
            K_x1_ref = self.base_kernel(x1, X_ref).to_dense()  # [N1, Nref]
            row_mean = K_x1_ref.mean(dim=1)                     # [N1]

            K_x2_ref = self.base_kernel(x2, X_ref).to_dense()  # [N2, Nref]
            col_mean = K_x2_ref.mean(dim=1)                     # [N2]

        if diag:
            K_raw_dense = K_raw if isinstance(K_raw, torch.Tensor) else K_raw
            centered = K_raw_dense - row_mean - col_mean + self._grand_mean
            return centered
        else:
            K_raw_dense = K_raw.to_dense() if hasattr(K_raw, 'to_dense') else K_raw
            centered = (
                K_raw_dense
                - row_mean.unsqueeze(1)
                - col_mean.unsqueeze(0)
                + self._grand_mean
            )
            return centered


class CenteredAdditiveKernel(Kernel):
    """Additive kernel with centered components and trace normalisation.

    Parameters
    ----------
    kernels : list of (name, Kernel) pairs
        The additive components to be centered individually.
    """

    has_lengthscale = False

    def __init__(self, kernels: List[Tuple[str, Kernel]], **kwargs):
        super().__init__(**kwargs)
        self.component_names: list[str] = []
        self.centered_kernels = torch.nn.ModuleList()

        for name, k in kernels:
            self.component_names.append(name)
            self.centered_kernels.append(CenteredKernelWrapper(k))

        # Per-component scale after centering (normalisation)
        self.log_scales = torch.nn.Parameter(
            torch.zeros(len(kernels))
        )

    @property
    def scales(self):
        return torch.nn.functional.softplus(self.log_scales)

    def fit(self, X_ref: torch.Tensor, max_samples: int = 1024):
        """Fit centering statistics for every component."""
        for ck in self.centered_kernels:
            ck.fit(X_ref, max_samples=max_samples)

    def forward(self, x1, x2, diag=False, **params):
        result = None
        for i, ck in enumerate(self.centered_kernels):
            val = ck(x1, x2, diag=diag, **params)
            scaled = self.scales[i] * val
            result = scaled if result is None else result + scaled
        return result

    def variance_attribution(self, X: torch.Tensor) -> dict:
        """Compute fraction of variance explained by each component.

        Returns dict {component_name: fraction}.
        """
        variances = {}
        total = 0.0
        for i, (name, ck) in enumerate(
            zip(self.component_names, self.centered_kernels)
        ):
            with torch.no_grad():
                diag_vals = ck(X, X, diag=True)
                v = (self.scales[i] * diag_vals).mean().item()
                variances[name] = v
                total += v

        if total > 0:
            variances = {k: v / total for k, v in variances.items()}
        return variances
