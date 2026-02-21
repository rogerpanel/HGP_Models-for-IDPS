"""
Multi-Scale Temporal Kernels  (Section IV.B, Eq. 33-35, Appendix D.4)
=====================================================================

Seven-component RBF bank (microsecond → week), three periodic components
(hourly / daily / weekly), optional spectral mixture for automatic pattern
discovery, and sigmoid change-point kernels for regime transitions.
"""

import math
import torch
import gpytorch
from gpytorch.kernels import (
    ScaleKernel,
    RBFKernel,
    PeriodicKernel,
    MaternKernel,
    SpectralMixtureKernel,
    Kernel,
)
from gpytorch.priors import LogNormalPrior


# ------------------------------------------------------------------ #
#  Change-Point Kernel (Appendix D.4)                                 #
# ------------------------------------------------------------------ #
class ChangePointKernel(Kernel):
    """Sigmoid-modulated kernel modelling regime transitions.

    k_cp(t, t') = σ(t - t_c) · k_post(t, t') + (1 - σ(t - t_c)) · k_pre(t, t')

    where σ is a sigmoid with learnable location t_c and steepness s.
    """

    has_lengthscale = False

    def __init__(self, pre_kernel: Kernel, post_kernel: Kernel, **kwargs):
        super().__init__(**kwargs)
        self.pre_kernel = pre_kernel
        self.post_kernel = post_kernel
        # Learnable change-point location and steepness
        self.register_parameter(
            "raw_changepoint", torch.nn.Parameter(torch.tensor(0.0))
        )
        self.register_parameter(
            "raw_steepness", torch.nn.Parameter(torch.tensor(1.0))
        )

    @property
    def changepoint(self):
        return self.raw_changepoint

    @property
    def steepness(self):
        return torch.nn.functional.softplus(self.raw_steepness)

    def forward(self, x1, x2, diag=False, **params):
        # Use last dimension as temporal index
        t1 = x1[..., -1:]
        t2 = x2[..., -1:]

        sig1 = torch.sigmoid(self.steepness * (t1 - self.changepoint))
        sig2 = torch.sigmoid(self.steepness * (t2 - self.changepoint))

        if diag:
            weight = (sig1.squeeze(-1) * sig2.squeeze(-1))
            k_post = self.post_kernel(x1, x2, diag=True, **params)
            k_pre = self.pre_kernel(x1, x2, diag=True, **params)
            return weight * k_post + (1 - weight) * k_pre

        weight = sig1 @ sig2.transpose(-2, -1)
        k_post = self.post_kernel(x1, x2, diag=False, **params)
        k_pre = self.pre_kernel(x1, x2, diag=False, **params)
        return weight * k_post.to_dense() + (1 - weight) * k_pre.to_dense()


# ------------------------------------------------------------------ #
#  Composite Multi-Scale Temporal Kernel                              #
# ------------------------------------------------------------------ #
class MultiScaleTemporalKernel(Kernel):
    """Additive composition of RBF bank + periodic + spectral + changepoint.

    Returns a list of sub-kernels that can be individually centered before
    being summed.

    Parameters
    ----------
    use_spectral : bool
        Include SpectralMixtureKernel for automatic periodicity discovery.
    num_mixtures : int
        Number of spectral-mixture components.
    use_changepoint : bool
        Include sigmoid change-point kernel.
    """

    has_lengthscale = False

    def __init__(
        self,
        use_spectral: bool = True,
        num_mixtures: int = 4,
        use_changepoint: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sub_kernels = torch.nn.ModuleList()
        self.sub_kernel_names: list[str] = []

        # --- 1. Seven-scale RBF bank (Eq. 33) ---
        rbf_scales = {
            "microsecond": -6,
            "millisecond": -3,
            "second": 0,
            "minute": 2,
            "hour": 3.6,
            "day": 4.9,
            "week": 5.8,
        }
        for name, log_scale in rbf_scales.items():
            k = ScaleKernel(
                RBFKernel(
                    lengthscale_prior=LogNormalPrior(
                        torch.tensor(float(log_scale)), torch.tensor(0.5)
                    )
                )
            )
            self.sub_kernels.append(k)
            self.sub_kernel_names.append(f"rbf_{name}")

        # --- 2. Periodic components (Eq. 34) ---
        for name, period in [("hourly", 3600), ("daily", 86400), ("weekly", 604800)]:
            k = ScaleKernel(
                PeriodicKernel(
                    period_length_prior=LogNormalPrior(
                        torch.tensor(math.log(period)), torch.tensor(0.1)
                    )
                )
            )
            self.sub_kernels.append(k)
            self.sub_kernel_names.append(f"periodic_{name}")

        # --- 3. Spectral mixture (Eq. 35) ---
        if use_spectral:
            k = SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=1)
            self.sub_kernels.append(k)
            self.sub_kernel_names.append("spectral_mixture")

        # --- 4. Change-point ---
        if use_changepoint:
            k = ChangePointKernel(
                pre_kernel=ScaleKernel(RBFKernel()),
                post_kernel=ScaleKernel(MaternKernel(nu=2.5)),
            )
            self.sub_kernels.append(ScaleKernel(k))
            self.sub_kernel_names.append("changepoint")

    # ------------------------------------------------------------------
    def forward(self, x1, x2, diag=False, **params):
        result = None
        for k in self.sub_kernels:
            val = k(x1, x2, diag=diag, **params)
            if diag:
                result = val if result is None else result + val
            else:
                result = val if result is None else result + val
        return result

    def get_sub_kernels(self):
        """Return list of (name, kernel) pairs for centering."""
        return list(zip(self.sub_kernel_names, self.sub_kernels))
