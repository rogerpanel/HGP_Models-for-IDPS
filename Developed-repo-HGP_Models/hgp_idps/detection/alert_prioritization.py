"""
Three-Tier Alert Prioritisation  (Section IV.D.3)
=================================================

Tier  |  σ threshold       |  Action
------+--------------------+---------
Low   |  σ < 1.0           |  log only
Med   |  1.0 ≤ σ < 2.0    |  queue for analyst
Crit  |  σ ≥ 2.0           |  page / escalate
"""

from dataclasses import dataclass
from typing import Dict, List
import torch


@dataclass
class Alert:
    sample_idx: int
    score: float
    confidence: float
    epistemic: float
    aleatoric: float
    tier: str           # "low", "medium", "critical"
    action: str         # "log", "queue", "page"


class AlertPrioritizer:
    """Map detector outputs to tiered alerts.

    Parameters
    ----------
    sigma_thresholds : tuple
        (low_max, med_max).  Defaults to (1.0, 2.0).
    """

    ACTIONS = {"low": "log", "medium": "queue", "critical": "page"}

    def __init__(self, sigma_thresholds=(1.0, 2.0)):
        self.lo, self.hi = sigma_thresholds

    def prioritize(self, detection_results: Dict[str, torch.Tensor]) -> List[Alert]:
        """Convert batch detection output into prioritised alert list."""
        scores = detection_results["scores"].cpu()
        conf = detection_results["confidence"].cpu()
        epist = detection_results["uncertainties"]["epistemic"].cpu()
        aleat = detection_results["uncertainties"]["aleatoric"].cpu()
        dets = detection_results["detections"].cpu()

        alerts: list[Alert] = []
        for i in range(len(scores)):
            if dets[i] < 0.5:
                continue  # not flagged
            s = scores[i].item()
            if s < self.lo:
                tier = "low"
            elif s < self.hi:
                tier = "medium"
            else:
                tier = "critical"

            alerts.append(Alert(
                sample_idx=i,
                score=s,
                confidence=conf[i].item(),
                epistemic=epist[i].item(),
                aleatoric=aleat[i].item(),
                tier=tier,
                action=self.ACTIONS[tier],
            ))

        return alerts

    @staticmethod
    def summary(alerts: List[Alert]) -> Dict[str, int]:
        counts = {"low": 0, "medium": 0, "critical": 0}
        for a in alerts:
            counts[a.tier] += 1
        return counts
