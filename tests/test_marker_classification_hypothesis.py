"""Property-based tests for xldvp_seg.analysis.marker_classification using Hypothesis.

Addresses Reviewer #3 gap: "No property-based tests". These tests verify invariants
that must hold for *any* valid input, not just hand-picked examples.
"""

import tempfile
from pathlib import Path

import hypothesis.extra.numpy as hnp
import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from xldvp_seg.analysis.marker_classification import (
    classify_gmm,
    classify_otsu,
    classify_otsu_half,
    classify_single_marker,
    extract_marker_values,
)

# ---------------------------------------------------------------------------
# Core classification functions: crash-freedom and output shape contracts
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(
    values=hnp.arrays(
        np.float64,
        st.integers(20, 500),
        elements=st.floats(0.01, 1000.0),
    )
)
def test_classify_otsu_never_crashes(values):
    """Otsu classification should never crash on any valid float array."""
    threshold, mask = classify_otsu(values)
    assert isinstance(threshold, float)
    assert len(mask) == len(values)
    assert mask.dtype == bool


@settings(max_examples=50)
@given(
    values=hnp.arrays(
        np.float64,
        st.integers(20, 500),
        elements=st.floats(0.01, 1000.0),
    )
)
def test_classify_otsu_half_never_crashes(values):
    """otsu_half classification should never crash on any valid float array."""
    threshold, mask = classify_otsu_half(values)
    assert isinstance(threshold, float)
    assert len(mask) == len(values)
    assert mask.dtype == bool


@settings(max_examples=50, deadline=None)
@given(
    values=hnp.arrays(
        np.float64,
        st.integers(20, 500),
        elements=st.floats(0.01, 1000.0),
    )
)
def test_classify_gmm_never_crashes(values):
    """GMM should handle any positive float array without crashing."""
    threshold, mask = classify_gmm(values)
    assert isinstance(threshold, float)
    assert len(mask) == len(values)
    assert mask.dtype == bool


# ---------------------------------------------------------------------------
# Relational properties: otsu_half is always more permissive than otsu
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(
    values=hnp.arrays(
        np.float64,
        st.integers(20, 500),
        elements=st.floats(0.01, 1000.0),
    )
)
def test_otsu_half_more_permissive(values):
    """otsu_half should classify >= as many cells as full otsu."""
    _, mask_full = classify_otsu(values)
    _, mask_half = classify_otsu_half(values)
    assert mask_half.sum() >= mask_full.sum()


@settings(max_examples=50)
@given(
    values=hnp.arrays(
        np.float64,
        st.integers(20, 500),
        elements=st.floats(0.01, 1000.0),
    )
)
def test_otsu_half_threshold_leq_full(values):
    """otsu_half threshold should be <= the full otsu threshold."""
    t_full, _ = classify_otsu(values)
    t_half, _ = classify_otsu_half(values)
    assert t_half <= t_full + 1e-9  # small epsilon for float comparison


# ---------------------------------------------------------------------------
# Threshold contracts: no positive below threshold, all positive >= threshold
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(
    values=hnp.arrays(
        np.float64,
        st.integers(20, 500),
        elements=st.floats(0.01, 1000.0),
    )
)
def test_otsu_positive_values_above_threshold(values):
    """Every Otsu-positive cell must have value >= threshold and > 0."""
    threshold, mask = classify_otsu(values)
    if mask.any():
        assert values[mask].min() >= threshold
        assert values[mask].min() > 0


@settings(max_examples=50)
@given(
    values=hnp.arrays(
        np.float64,
        st.integers(20, 500),
        elements=st.floats(0.01, 1000.0),
    )
)
def test_otsu_half_positive_values_above_threshold(values):
    """Every otsu_half-positive cell must have value >= threshold and > 0."""
    threshold, mask = classify_otsu_half(values)
    if mask.any():
        assert values[mask].min() >= threshold
        assert values[mask].min() > 0


# ---------------------------------------------------------------------------
# Edge cases: constant arrays
# ---------------------------------------------------------------------------


@settings(max_examples=20)
@given(
    val=st.floats(0.01, 1000.0),
    n=st.integers(20, 200),
)
def test_otsu_constant_array_all_negative(val, n):
    """Constant (zero-variance) arrays should yield all-negative classification."""
    values = np.full(n, val)
    threshold, mask = classify_otsu(values)
    # Zero variance triggers the safety guard -> threshold=0, all negative
    assert threshold == 0.0
    assert mask.sum() == 0


@settings(max_examples=20, deadline=None)
@given(
    val=st.floats(0.01, 1000.0),
    n=st.integers(20, 200),
)
def test_gmm_constant_array_all_negative(val, n):
    """Constant (zero-variance) arrays should yield all-negative GMM classification."""
    values = np.full(n, val)
    threshold, mask = classify_gmm(values)
    assert threshold == 0.0
    assert mask.sum() == 0


# ---------------------------------------------------------------------------
# SNR threshold monotonicity
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    snr_threshold_low=st.floats(0.5, 4.9),
    snr_threshold_high=st.floats(5.0, 10.0),
)
def test_snr_threshold_monotonic(snr_threshold_low, snr_threshold_high):
    """Higher SNR threshold should classify fewer (or equal) cells as positive."""
    assume(snr_threshold_low < snr_threshold_high)
    np.random.seed(42)
    values = np.concatenate([np.random.normal(1, 0.3, 200), np.random.normal(5, 0.5, 200)])
    values = np.maximum(values, 0.0)

    dets_low = [
        {
            "uid": f"c{i}",
            "global_center": [0.0, 0.0],
            "features": {"ch1_snr": float(v)},
        }
        for i, v in enumerate(values)
    ]
    dets_high = [
        {
            "uid": f"c{i}",
            "global_center": [0.0, 0.0],
            "features": {"ch1_snr": float(v)},
        }
        for i, v in enumerate(values)
    ]

    tmp_dir = Path(tempfile.mkdtemp())
    out_low = tmp_dir / "low"
    out_low.mkdir()
    out_high = tmp_dir / "high"
    out_high.mkdir()

    result_low = classify_single_marker(
        dets_low,
        channel=1,
        marker_name="Test",
        method="snr",
        output_dir=out_low,
        snr_threshold=snr_threshold_low,
    )
    result_high = classify_single_marker(
        dets_high,
        channel=1,
        marker_name="Test",
        method="snr",
        output_dir=out_high,
        snr_threshold=snr_threshold_high,
    )

    assert result_low["n_positive"] >= result_high["n_positive"]
    assert result_low["n_positive"] + result_low["n_negative"] == len(values)
    assert result_high["n_positive"] + result_high["n_negative"] == len(values)


# ---------------------------------------------------------------------------
# classify_single_marker: output contract
# ---------------------------------------------------------------------------


@settings(max_examples=30, deadline=None)
@given(
    n=st.integers(30, 300),
    method=st.sampled_from(["otsu", "otsu_half", "gmm"]),
)
def test_classify_single_marker_output_contract(n, method):
    """classify_single_marker must return well-formed summary for any method."""
    np.random.seed(42)
    values = np.concatenate(
        [np.random.normal(2.0, 0.5, n // 2), np.random.normal(10.0, 1.0, n - n // 2)]
    )
    values = np.maximum(values, 0.01)

    dets = [
        {
            "uid": f"cell_{i}",
            "global_center": [float(i * 10), float(i * 10)],
            "features": {"ch1_mean": float(v)},
        }
        for i, v in enumerate(values)
    ]

    out_dir = Path(tempfile.mkdtemp()) / f"{method}_{n}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = classify_single_marker(
        dets,
        channel=1,
        marker_name="PropTest",
        method=method,
        output_dir=out_dir,
    )

    # Summary dict has required keys
    assert "n_positive" in result
    assert "n_negative" in result
    assert "threshold" in result
    assert isinstance(result["threshold"], float)

    # Conservation: all cells accounted for
    assert result["n_positive"] + result["n_negative"] == n
    assert result["n_positive"] >= 0
    assert result["n_negative"] >= 0

    # Every detection was mutated in-place
    for det in dets:
        assert "PropTest_class" in det["features"]
        assert det["features"]["PropTest_class"] in ("positive", "negative")
        assert "PropTest_value" in det["features"]
        assert isinstance(det["features"]["PropTest_value"], float)


# ---------------------------------------------------------------------------
# extract_marker_values: length invariant
# ---------------------------------------------------------------------------


@settings(max_examples=30)
@given(n=st.integers(1, 200))
def test_extract_marker_values_length(n):
    """Extracted values array must match detection list length."""
    dets = [{"uid": f"d{i}", "features": {"ch0_mean": float(i)}} for i in range(n)]
    vals = extract_marker_values(dets, channel=0, feature="mean")
    assert len(vals) == n
    assert vals.dtype == np.float64
