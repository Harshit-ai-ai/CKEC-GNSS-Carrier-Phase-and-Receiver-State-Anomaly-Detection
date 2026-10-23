
import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_real_data_experimental import (
    _benign_stats,
    strict_chronological_split,
)


def test_calibration_stats_ignore_all_post_calibration_data():
    labels = np.array([0] * 5 + [1] * 3)
    original = pd.Series([1., 2., 3., 4., 5., 100., 200., 300.])
    changed = pd.Series([1., 2., 3., 4., 5., -9000., 8000., 7000.])

    mu1, sd1 = _benign_stats(original, labels, 5)
    mu2, sd2 = _benign_stats(changed, np.zeros(8, dtype=int), 5)

    assert mu1 == pytest.approx(mu2)
    assert sd1 == pytest.approx(sd2)


def test_calibration_stats_reject_attack_in_initial_interval():
    values = pd.Series([1., 2., 3., 4., 5., 6., 7., 8.])
    labels = np.array([0, 0, 1, 0, 0, 1, 1, 1])

    with pytest.raises(ValueError, match="calibration"):
        _benign_stats(values, labels, 5)


def test_calibration_stats_reject_insufficient_finite_values():
    values = pd.Series([1., np.nan, 3., np.inf, 5., 6.])
    labels = np.zeros(6, dtype=int)

    with pytest.raises(ValueError, match="finite observations"):
        _benign_stats(values, labels, 5)


def test_strict_split_uses_initial_benign_prefix():
    # First 300 rows are benign; attack occurs from row 300 through 699;
    # the final 300 rows are benign.
    labels = np.zeros(1000, dtype=int)
    labels[300:700] = 1
    df = pd.DataFrame({"attack_label": labels})

    cal, val, test = strict_chronological_split(df)

    # Default calibration fraction is 0.67 of the 300-row benign prefix.
    assert len(cal) == 201
    assert len(val) == 99
    assert (cal[0], cal[-1]) == (0, 200)
    assert (val[0], val[-1]) == (201, 299)
    assert (test[0], test[-1]) == (300, 999)

    assert cal[-1] < val[0] < val[-1] < test[0]
    assert np.all(labels[cal] == 0)
    assert np.all(labels[val] == 0)
    assert np.any(labels[test] == 0)
    assert np.any(labels[test] != 0)


def test_strict_split_rejects_short_benign_prefix():
    labels = np.zeros(1000, dtype=int)
    labels[120:900] = 1
    df = pd.DataFrame({"attack_label": labels})

    with pytest.raises(ValueError, match="prefix too short"):
        strict_chronological_split(df)


def test_strict_split_rejects_test_without_benign_samples():
    # The initial prefix is long enough, but every row after the first
    # attack is an attack. There is no benign test sample.
    labels = np.zeros(1000, dtype=int)
    labels[300:] = 1
    df = pd.DataFrame({"attack_label": labels})

    with pytest.raises(ValueError, match="both benign and attack"):
        strict_chronological_split(df)


def test_strict_split_rejects_scenario_without_attacks():
    labels = np.zeros(1000, dtype=int)
    df = pd.DataFrame({"attack_label": labels})

    with pytest.raises(ValueError, match="no attack"):
        strict_chronological_split(df)


def test_strict_split_rejects_empty_scenario():
    df = pd.DataFrame({"attack_label": np.array([], dtype=int)})

    with pytest.raises(ValueError, match="empty"):
        strict_chronological_split(df)


def test_prepare_features_preserves_original_row_count_and_labels():
    from scripts.evaluate_real_data_experimental import prepare_features

    n = 120
    x = np.arange(n, dtype=float)
    df = pd.DataFrame({
        "time_s": x,
        "carrier_phase": (
            x + 0.03 * np.sin(x * 0.37) + 0.01 * np.cos(x * 0.13)
        ),
        "hardware_agc": 20.0 + 2.0 * np.sin(x * 0.17),
        "cn0_db_hz": 40.0 + 2.0 * np.sin(x * 0.11),
        "attack_label": np.zeros(n, dtype=int),
    })

    features = prepare_features(df, calibration_n=40)

    assert len(features) == len(df)
    np.testing.assert_array_equal(
        features["attack_label"].to_numpy(),
        df["attack_label"].to_numpy(),
    )
    np.testing.assert_array_equal(
        features["time_s"].to_numpy(),
        df["time_s"].to_numpy(),
    )


def test_prepare_features_rejects_attack_in_calibration_interval():
    from scripts.evaluate_real_data_experimental import prepare_features

    n = 120
    labels = np.zeros(n, dtype=int)
    labels[10] = 1
    df = pd.DataFrame({
        "time_s": np.arange(n, dtype=float),
        "carrier_phase": np.arange(n, dtype=float),
        "attack_label": labels,
    })

    with pytest.raises(ValueError, match="calibration"):
        prepare_features(df, calibration_n=40)
