from setuptools import setup, find_packages

setup(
    name="hgp-idps",
    version="3.0.0",
    description=(
        "Uncertainty-Calibrated Hierarchical Gaussian Processes "
        "for Intrusion Detection with Multi-Scale Temporal Modeling"
    ),
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "gpytorch>=1.11",
        "numpy>=1.24.3",
        "pandas>=2.0.0",
        "scipy>=1.11.1",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "kagglehub>=0.2.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
