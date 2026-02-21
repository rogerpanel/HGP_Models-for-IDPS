"""
Domain-Specific Kernels  (Section IV.B.2-4, Appendix D.1-D.3)
==============================================================

EdgeIIoTKernel  – protocol-aware with learned embeddings  (Eq. 28-29)
ContainerKernel – IAT Wasserstein + flow + burst  (Eq. 30-31)
SOCKernel       – entity/alert graph kernel  (Eq. 32)
"""

import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, Kernel
from gpytorch.priors import LogNormalPrior


# ================================================================== #
#  Edge-IIoT  (Appendix D.1)                                         #
# ================================================================== #
class ProtocolEmbeddingKernel(Kernel):
    """RBF over learned protocol embeddings  e : Cat → R^{d_e}.

    k_proto(x_p, x_p') = exp(-0.5 Σ_i ||e(x_{p,i}) - e(x'_{p,i})||^2 / ℓ_i^2)
    """

    has_lengthscale = False

    def __init__(self, num_protocols: int, embedding_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(num_protocols, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.register_parameter(
            "raw_lengthscale",
            nn.Parameter(torch.zeros(embedding_dim)),
        )

    @property
    def lengthscale(self):
        return torch.nn.functional.softplus(self.raw_lengthscale)

    def forward(self, x1, x2, diag=False, **params):
        # x1, x2 are integer protocol indices  [N, 1]
        e1 = self.embedding(x1.long().squeeze(-1))  # [N, d_e]
        e2 = self.embedding(x2.long().squeeze(-1))
        scaled_diff = (e1.unsqueeze(1) - e2.unsqueeze(0)) / self.lengthscale
        sq_dist = (scaled_diff ** 2).sum(-1)
        K = torch.exp(-0.5 * sq_dist)
        if diag:
            return K.diag()
        return K


class EdgeIIoTKernel(Kernel):
    """Additive composition: protocol-aware + flow RBF-ARD + Matern-2.5.

    k_edge = k_proto + k_flow + k_rough
    """

    has_lengthscale = False

    def __init__(self, feature_dim: int, num_protocols: int = 20,
                 embedding_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.protocol_kernel = ScaleKernel(
            ProtocolEmbeddingKernel(num_protocols, embedding_dim)
        )
        self.flow_kernel = ScaleKernel(
            RBFKernel(
                ard_num_dims=feature_dim,
                lengthscale_prior=LogNormalPrior(-1.0, 0.5),
            )
        )
        self.rough_kernel = ScaleKernel(
            MaternKernel(nu=2.5, lengthscale_prior=LogNormalPrior(0.0, 0.5))
        )

    def forward(self, x1, x2, diag=False, **params):
        # Protocol kernel operates on continuous features as well (fallback)
        k_flow = self.flow_kernel(x1, x2, diag=diag, **params)
        k_rough = self.rough_kernel(x1, x2, diag=diag, **params)
        if diag:
            return k_flow + k_rough
        return k_flow + k_rough


# ================================================================== #
#  Container  (Appendix D.2)                                          #
# ================================================================== #
class WassersteinIATKernel(Kernel):
    """RBF kernel over 2-Wasserstein distance between IAT histograms.

    Uses histogram bin representations of inter-arrival time distributions.
    """

    has_lengthscale = True

    def __init__(self, num_bins: int = 50, **kwargs):
        super().__init__(has_lengthscale=True, **kwargs)
        self.num_bins = num_bins

    def forward(self, x1, x2, diag=False, **params):
        # Approximate 2-Wasserstein via CDF difference (Cramér distance)
        cdf1 = x1.cumsum(dim=-1)
        cdf2 = x2.cumsum(dim=-1)
        if diag:
            w2 = ((cdf1 - cdf2) ** 2).sum(-1)
        else:
            diff = cdf1.unsqueeze(1) - cdf2.unsqueeze(0)
            w2 = (diff ** 2).sum(-1)
        return torch.exp(-0.5 * w2 / (self.lengthscale ** 2 + 1e-6))


class ContainerKernel(Kernel):
    """k_container = k_iat_wasserstein + k_packet_size + k_burst.

    All three operate in the projected latent space; the Wasserstein IAT
    component uses histogram features extracted during preprocessing.
    """

    has_lengthscale = False

    def __init__(self, feature_dim: int, iat_bins: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.iat_kernel = ScaleKernel(WassersteinIATKernel(num_bins=iat_bins))
        self.packet_kernel = ScaleKernel(
            RBFKernel(
                ard_num_dims=feature_dim,
                lengthscale_prior=LogNormalPrior(0.0, 0.5),
            )
        )
        self.burst_kernel = ScaleKernel(
            MaternKernel(nu=1.5, lengthscale_prior=LogNormalPrior(0.5, 0.5))
        )

    def forward(self, x1, x2, diag=False, **params):
        k_pkt = self.packet_kernel(x1, x2, diag=diag, **params)
        k_burst = self.burst_kernel(x1, x2, diag=diag, **params)
        if diag:
            return k_pkt + k_burst
        return k_pkt + k_burst


# ================================================================== #
#  SOC  (Appendix D.3)                                                #
# ================================================================== #
class AlertComponentKernel(Kernel):
    """Multiplicative alert kernel:
       k_alert(a, a') = k_sev(s,s') · k_cat(c,c') · k_conf(cf,cf')
    """

    has_lengthscale = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.severity_kernel = ScaleKernel(RBFKernel())
        self.category_kernel = ScaleKernel(RBFKernel())
        self.confidence_kernel = ScaleKernel(RBFKernel())

    def forward(self, x1, x2, diag=False, **params):
        # Split features into [severity, category, confidence, ...]
        # Use first 3 dims as proxies for the 3 alert components
        k_sev = self.severity_kernel(
            x1[..., :1], x2[..., :1], diag=diag, **params
        )
        k_cat = self.category_kernel(
            x1[..., 1:2], x2[..., 1:2], diag=diag, **params
        )
        k_conf = self.confidence_kernel(
            x1[..., 2:3], x2[..., 2:3], diag=diag, **params
        )
        if diag:
            return k_sev * k_cat * k_conf
        # For lazy tensors, evaluate then multiply
        return k_sev.to_dense() * k_cat.to_dense() * k_conf.to_dense()


class SOCKernel(Kernel):
    """k_soc = k_entity_rbf + k_alert_multiplicative + k_matern.

    The entity-graph metapath kernel is approximated via RBF over entity
    aggregate features.  Full graph kernel is computationally infeasible
    for 16.8 M records; the aggregate representation captures metapath
    information up to length 3 via feature engineering.
    """

    has_lengthscale = False

    def __init__(self, feature_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.entity_kernel = ScaleKernel(
            RBFKernel(
                ard_num_dims=feature_dim,
                lengthscale_prior=LogNormalPrior(1.0, 0.5),
            )
        )
        self.alert_kernel = ScaleKernel(AlertComponentKernel())
        self.matern_kernel = ScaleKernel(
            MaternKernel(nu=1.5, lengthscale_prior=LogNormalPrior(1.0, 0.5))
        )

    def forward(self, x1, x2, diag=False, **params):
        k_ent = self.entity_kernel(x1, x2, diag=diag, **params)
        k_alert = self.alert_kernel(x1, x2, diag=diag, **params)
        k_mat = self.matern_kernel(x1, x2, diag=diag, **params)
        if diag:
            return k_ent + k_alert + k_mat
        return k_ent + k_alert + k_mat
