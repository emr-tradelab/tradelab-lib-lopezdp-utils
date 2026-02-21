"""Portfolio allocation — HRP, denoising, NCO — AFML Ch. 16 + MLAM Sections 2, 7.

This package implements López de Prado's portfolio construction methods:
- Hierarchical Risk Parity (HRP): graph-theory-based allocation avoiding
  covariance matrix inversion
- Random Matrix Theory denoising: Marcenko-Pastur-based cleaning of empirical
  covariance matrices
- Nested Clustered Optimization (NCO): hierarchical optimization with ONC
  clustering from features/

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapter 16
    Lopez de Prado, "Machine Learning for Asset Managers", Sections 2.4-2.9, 7.3-7.6
"""

from tradelab.lopezdp_utils.allocation.denoising import (
    denoise_cov,
    denoised_corr,
    denoised_corr_shrinkage,
    detone_corr,
    find_max_eval,
    mp_pdf,
)
from tradelab.lopezdp_utils.allocation.hrp import (
    correl_dist,
    get_cluster_var,
    get_ivp,
    get_quasi_diag,
    get_rec_bipart,
    hrp_alloc,
    tree_clustering,
)
from tradelab.lopezdp_utils.allocation.nco import opt_port_nco
from tradelab.lopezdp_utils.allocation.simulation import generate_data, hrp_mc

__all__ = [
    "correl_dist",
    "denoise_cov",
    "denoised_corr",
    "denoised_corr_shrinkage",
    "detone_corr",
    "find_max_eval",
    "generate_data",
    "get_cluster_var",
    "get_ivp",
    "get_quasi_diag",
    "get_rec_bipart",
    "hrp_alloc",
    "hrp_mc",
    "mp_pdf",
    "opt_port_nco",
    "tree_clustering",
]
