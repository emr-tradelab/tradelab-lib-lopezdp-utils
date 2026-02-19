"""Feature engineering — AFML Chapters 5, 8, 17-18 + MLAM Chapters 3-4, 6.

This package covers the fourth stage of López de Prado's pipeline:
labeled data → engineered features (stationarity, entropy, structural breaks,
importance analysis, orthogonalization).

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapters 5, 8, 17-18
    López de Prado, "Machine Learning for Asset Managers", Chapters 3-4, 6
"""

from tradelab.lopezdp_utils.features.entropy import (
    adverse_selection_feature,
    cross_entropy,
    discretize_optimal,
    encode_binary,
    encode_quantile,
    encode_sigma,
    kl_divergence,
    konto,
    lempel_ziv_lib,
    market_efficiency_metric,
    mutual_information_optimal,
    num_bins,
    plug_in,
    portfolio_concentration,
    variation_of_information,
)
from tradelab.lopezdp_utils.features.fractional_diff import (
    frac_diff,
    frac_diff_ffd,
    get_weights,
    get_weights_ffd,
    plot_min_ffd,
)
from tradelab.lopezdp_utils.features.importance import (
    cluster_kmeans_base,
    cluster_kmeans_top,
    feat_imp_mda,
    feat_imp_mda_clustered,
    feat_imp_mdi,
    feat_imp_mdi_clustered,
    feat_imp_sfi,
    feat_importance,
    get_test_data,
)
from tradelab.lopezdp_utils.features.orthogonal import (
    get_ortho_feats,
    pca_weights,
    weighted_kendall_tau,
)
from tradelab.lopezdp_utils.features.structural_breaks import (
    brown_durbin_evans_cusum,
    cadf_test,
    chow_type_dickey_fuller,
    chu_stinchcombe_white_cusum,
    get_betas,
    qadf_test,
    sadf_test,
)

__all__ = [
    # Fractional differentiation
    "get_weights",
    "get_weights_ffd",
    "frac_diff",
    "frac_diff_ffd",
    "plot_min_ffd",
    # Entropy & information theory
    "plug_in",
    "lempel_ziv_lib",
    "konto",
    "encode_binary",
    "encode_quantile",
    "encode_sigma",
    "market_efficiency_metric",
    "portfolio_concentration",
    "adverse_selection_feature",
    "kl_divergence",
    "cross_entropy",
    "num_bins",
    "discretize_optimal",
    "variation_of_information",
    "mutual_information_optimal",
    # Structural breaks
    "sadf_test",
    "brown_durbin_evans_cusum",
    "chu_stinchcombe_white_cusum",
    "chow_type_dickey_fuller",
    "qadf_test",
    "cadf_test",
    "get_betas",
    # Feature importance
    "feat_imp_mdi",
    "feat_imp_mda",
    "feat_imp_sfi",
    "cluster_kmeans_base",
    "cluster_kmeans_top",
    "feat_imp_mdi_clustered",
    "feat_imp_mda_clustered",
    "get_test_data",
    "feat_importance",
    # Orthogonal features
    "get_ortho_feats",
    "pca_weights",
    "weighted_kendall_tau",
]
