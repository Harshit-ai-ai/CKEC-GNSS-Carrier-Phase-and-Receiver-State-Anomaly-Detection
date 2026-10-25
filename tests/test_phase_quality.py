import numpy as np
import pandas as pd
import pytest

from scripts.phase_quality import flag_phase_discontinuities


def test_flags_large_jump_within_same_satellite():
    df = pd.DataFrame({
        "time": pd.to_datetime([
            "2024-09-10T12:00:00Z",
            "2024-09-10T12:00:01Z",
            "2024-09-10T12:00:02Z",
        ]),
        "satellite": ["C06", "C06", "C06"],
        "carrier_phase": [1000.0, 1001.0, -668.0],
    })

    flags = flag_phase_discontinuities(
        df, max_phase_rate_cycles_per_second=100
    )

    assert flags.tolist() == [False, False, True]
    # The input measurements remain untouched.
    assert df.loc[2, "carrier_phase"] == -668.0


def test_does_not_compare_different_satellites():
    df = pd.DataFrame({
        "time": pd.to_datetime([
            "2024-09-10T12:00:00Z",
            "2024-09-10T12:00:01Z",
        ]),
        "satellite": ["C06", "C07"],
        "carrier_phase": [1000.0, -668.0],
    })

    flags = flag_phase_discontinuities(
        df, max_phase_rate_cycles_per_second=100
    )
    assert not flags.any()

def test_flags_only_later_sample_of_phase_jump():
    df = pd.DataFrame({
        "time": pd.to_datetime([
            "2024-09-10T12:00:00Z",
            "2024-09-10T12:00:01Z",
            "2024-09-10T12:00:02Z",
        ]),
        "satellite": ["C06"] * 3,
        "carrier_phase": [1000.0, 1001.0, -668.0],
    })

    flags = flag_phase_discontinuities(
        df, max_phase_rate_cycles_per_second=100
    )

    assert flags.tolist() == [False, False, True]


def test_invalid_phase_is_not_modified_by_validator():
    df = pd.DataFrame({
        "time": pd.to_datetime([
            "2024-09-10T12:00:00Z",
            "2024-09-10T12:00:01Z",
        ]),
        "satellite": ["C06", "C06"],
        "carrier_phase": [1000.0, -668.0],
    })
    original = df["carrier_phase"].copy()

    flag_phase_discontinuities(
        df, max_phase_rate_cycles_per_second=100
    )

    pd.testing.assert_series_equal(df["carrier_phase"], original)


def test_ignores_gaps_larger_than_limit():
    df = pd.DataFrame({
        "time": pd.to_datetime([
            "2024-09-10T12:00:00Z",
            "2024-09-10T12:00:05Z",
        ]),
        "satellite": ["C06", "C06"],
        "carrier_phase": [1000.0, 9000.0],
    })

    flags = flag_phase_discontinuities(df, max_gap_seconds=2)
    assert not flags.any()


def test_handles_missing_phase():
    df = pd.DataFrame({
        "time": pd.to_datetime([
            "2024-09-10T12:00:00Z",
            "2024-09-10T12:00:01Z",
            "2024-09-10T12:00:02Z",
        ]),
        "satellite": ["C06", "C06", "C06"],
        "carrier_phase": [1000.0, np.nan, 1002.0],
    })

    flags = flag_phase_discontinuities(df)
    assert not flags.any()


def test_rejects_invalid_threshold():
    df = pd.DataFrame({
        "time": pd.to_datetime(["2024-09-10T12:00:00Z"]),
        "satellite": ["C06"],
        "carrier_phase": [1000.0],
    })

    with pytest.raises(ValueError, match="max_gap_seconds"):
        flag_phase_discontinuities(df, max_gap_seconds=0)
