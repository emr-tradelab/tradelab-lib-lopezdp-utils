"""Machine Learning Asset Allocation — AFML Chapter 16 + MLAM.

Hierarchical Risk Parity (HRP), covariance matrix denoising/detoning,
and Nested Clustered Optimization (NCO) for robust portfolio construction.
"""

from tradelab.lopezdp_utils.ml_asset_allocation.denoising import (
    denoise_cov,
    denoised_corr,
    denoised_corr_shrinkage,
    detone_corr,
    find_max_eval,
    mp_pdf,
)
from tradelab.lopezdp_utils.ml_asset_allocation.hrp import (
    correl_dist,
    get_cluster_var,
    get_ivp,
    get_quasi_diag,
    get_rec_bipart,
    hrp_alloc,
    tree_clustering,
)
from tradelab.lopezdp_utils.ml_asset_allocation.nco import opt_port_nco
from tradelab.lopezdp_utils.ml_asset_allocation.simulation import (
    generate_data,
    hrp_mc,
)

__all__ = [
    # HRP (AFML Ch.16)
    "correl_dist",
    "tree_clustering",
    "get_quasi_diag",
    "get_ivp",
    "get_cluster_var",
    "get_rec_bipart",
    "hrp_alloc",
    # Denoising/Detoning (MLAM Ch.2)
    "mp_pdf",
    "find_max_eval",
    "denoised_corr",
    "denoised_corr_shrinkage",
    "denoise_cov",
    "detone_corr",
    # NCO (MLAM Ch.7)
    "opt_port_nco",
    # Simulation (AFML Ch.16)
    "generate_data",
    "hrp_mc",
]
