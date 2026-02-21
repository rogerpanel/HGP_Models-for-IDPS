"""
ICS3D Dataset Loader  (Section VI, Appendix I)
==============================================

Downloads and loads the Integrated Cloud Security 3-Datasets:
  - Edge-IIoT  :  2.219 M samples,  60-140 protocol features
  - Container  :  234,560 samples, 87 flow-level features
  - SOC (GUIDE):  16.95 M samples, 33 entity types

Kaggle DOI: https://doi.org/10.34740/kaggle/dsv/12483891
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ICS3DDatasetLoader:
    """Download and load each ICS3D sub-dataset.

    Parameters
    ----------
    base_path : str or None
        If supplied, skip kagglehub download and use this directory.
    """

    DATASET_CONFIGS = {
        "edge_dnn": {
            "file": "DNN-EdgeIIoT-dataset.csv",
            "label_col": "Attack_type",
            "domain": "edge_iiot",
        },
        "edge_ml": {
            "file": "ML-EdgeIIoT-dataset.csv",
            "label_col": "Attack_type",
            "domain": "edge_iiot",
        },
        "containers": {
            "file": "Containers_Dataset.csv",
            "label_col": "Label",
            "domain": "container",
        },
        "soc_train": {
            "file": "Microsoft_GUIDE_Train.csv",
            "label_col": "IncidentGrade",
            "domain": "soc",
        },
        "soc_test": {
            "file": "Microsoft_GUIDE_Test.csv",
            "label_col": "IncidentGrade",
            "domain": "soc",
        },
    }

    def __init__(self, base_path: Optional[str] = None):
        if base_path is not None:
            self.base_path = Path(base_path)
        else:
            self.base_path = self._download()
        self.raw: Dict[str, pd.DataFrame] = {}

    @staticmethod
    def _download() -> Path:
        import kagglehub

        print("Downloading ICS3D dataset from Kaggle …")
        path = kagglehub.dataset_download(
            "rogernickanaedevha/integrated-cloud-security-3datasets-ics3d"
        )
        print(f"  Downloaded to {path}")
        return Path(path)

    def load(self, names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load specified (or all) sub-datasets into memory."""
        names = names or list(self.DATASET_CONFIGS.keys())
        print("\n" + "=" * 60)
        print("LOADING ICS3D DATASETS")
        print("=" * 60)

        for name in names:
            cfg = self.DATASET_CONFIGS.get(name)
            if cfg is None:
                print(f"  Unknown dataset: {name}")
                continue
            fp = self.base_path / cfg["file"]
            if not fp.exists():
                print(f"  {cfg['file']} not found — skipping")
                continue

            print(f"\n  Loading {name} …")
            df = pd.read_csv(fp, low_memory=False)
            print(f"    shape = {df.shape}")
            if cfg["label_col"] in df.columns:
                print(f"    unique labels = {df[cfg['label_col']].nunique()}")
            self.raw[name] = df

        return self.raw
