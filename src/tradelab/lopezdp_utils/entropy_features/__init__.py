"""Entropy Features — AFML Chapter 18 + MLAM Chapter 3.

Entropy estimators, encoding schemes, and information-theoretic metrics for
quantifying market efficiency, portfolio concentration, and adverse selection.
"""

from tradelab.lopezdp_utils.entropy_features.applications import (
    adverse_selection_feature,
    market_efficiency_metric,
    portfolio_concentration,
)
from tradelab.lopezdp_utils.entropy_features.encoding import (
    encode_binary,
    encode_quantile,
    encode_sigma,
)
from tradelab.lopezdp_utils.entropy_features.estimators import (
    konto,
    lempel_ziv_lib,
    match_length,
    plug_in,
    pmf1,
)
from tradelab.lopezdp_utils.entropy_features.information_theory import (
    cross_entropy,
    kl_divergence,
)

__all__ = [
    "adverse_selection_feature",
    "cross_entropy",
    "encode_binary",
    "encode_quantile",
    "encode_sigma",
    "kl_divergence",
    "konto",
    "lempel_ziv_lib",
    "market_efficiency_metric",
    "match_length",
    "plug_in",
    "pmf1",
    "portfolio_concentration",
]
