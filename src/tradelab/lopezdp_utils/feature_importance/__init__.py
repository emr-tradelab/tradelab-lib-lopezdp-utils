"""Feature Importance — AFML Chapter 8.

Framework for understanding the variables that contribute to a model's
performance. Includes three complementary importance methods (MDI, MDA, SFI),
PCA-based orthogonalization, consistency validation, and a synthetic data
testing suite. Extended with Clustered Feature Importance from ML for Asset
Managers to address multicollinearity via ONC clustering.
"""

from tradelab.lopezdp_utils.feature_importance.clustering import (
    cluster_kmeans_base,
    cluster_kmeans_top,
    feat_imp_mda_clustered,
    feat_imp_mdi_clustered,
)
from tradelab.lopezdp_utils.feature_importance.importance import (
    feat_imp_mda,
    feat_imp_mdi,
    feat_imp_sfi,
)
from tradelab.lopezdp_utils.feature_importance.orthogonal import (
    get_e_vec,
    get_ortho_feats,
    weighted_kendall_tau,
)
from tradelab.lopezdp_utils.feature_importance.synthetic import (
    feat_importance,
    get_test_data,
    plot_feat_importance,
    test_func,
)

__all__ = [
    "cluster_kmeans_base",
    "cluster_kmeans_top",
    "feat_imp_mda",
    "feat_imp_mda_clustered",
    "feat_imp_mdi",
    "feat_imp_mdi_clustered",
    "feat_imp_sfi",
    "feat_importance",
    "get_e_vec",
    "get_ortho_feats",
    "get_test_data",
    "plot_feat_importance",
    "test_func",
    "weighted_kendall_tau",
]
