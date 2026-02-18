"""Tests for labeling.class_balance."""

import numpy as np
import polars as pl


class TestDropLabels:
    def test_drops_rare_class(self):
        from tradelab.lopezdp_utils.labeling.class_balance import drop_labels

        events = pl.DataFrame(
            {
                "label": [1] * 90 + [0] * 5 + [-1] * 5,
                "ret": np.random.randn(100),
            }
        )
        result = drop_labels(events, min_pct=0.08)
        # Classes with < 8% should be dropped
        remaining = result["label"].unique().to_list()
        assert 1 in remaining
        # 0 and -1 are each 5% < 8%, should be dropped
        assert 0 not in remaining
        assert -1 not in remaining

    def test_keeps_all_when_above_threshold(self):
        from tradelab.lopezdp_utils.labeling.class_balance import drop_labels

        events = pl.DataFrame(
            {
                "label": [1] * 40 + [0] * 30 + [-1] * 30,
            }
        )
        result = drop_labels(events, min_pct=0.05)
        assert len(result["label"].unique()) == 3


class TestGetClassWeights:
    def test_returns_dict(self):
        from tradelab.lopezdp_utils.labeling.class_balance import get_class_weights

        labels = pl.Series("label", [1, 1, 1, 0, 0, -1])
        result = get_class_weights(labels)
        assert isinstance(result, dict)
        assert -1 in result
        assert result[-1] > result[1]  # minority class gets higher weight
