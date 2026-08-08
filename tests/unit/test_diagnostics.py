import numpy as np
import pytest

from thermo_lab.diagnostics import diagnose_series, summarize_chain


def test_iid_series_has_ess_near_recorded_length() -> None:
    series = np.random.default_rng(1234).normal(size=4_000)

    diagnostic = diagnose_series(series)

    assert diagnostic.status == "estimated"
    assert diagnostic.ess is not None
    assert 2_000 <= diagnostic.ess <= 4_000


def test_positive_ar1_correlation_reduces_ess() -> None:
    rng = np.random.default_rng(99)
    noise = rng.normal(size=4_000)
    series = np.empty_like(noise)
    series[0] = noise[0]
    for index in range(1, len(series)):
        series[index] = 0.85 * series[index - 1] + noise[index]

    diagnostic = diagnose_series(series)

    assert diagnostic.ess is not None
    assert diagnostic.ess < 1_000
    assert diagnostic.lag_1_autocorrelation == pytest.approx(0.85, abs=0.05)


def test_constant_series_is_explicitly_uninformative() -> None:
    diagnostic = diagnose_series(np.ones(100))

    assert diagnostic.status == "constant_series"
    assert diagnostic.ess == 0.0
    assert diagnostic.integrated_autocorrelation_time is None
    assert diagnostic.lag_1_autocorrelation is None


@pytest.mark.parametrize("length", [0, 1, 2, 3])
def test_short_series_does_not_report_precision(length: int) -> None:
    diagnostic = diagnose_series(np.arange(length, dtype=float))

    assert diagnostic.status == "insufficient_length"
    assert diagnostic.ess is None
    assert diagnostic.integrated_autocorrelation_time is None


def test_ess_never_exceeds_recorded_state_count() -> None:
    for seed in range(10):
        series = np.random.default_rng(seed).normal(size=200)
        diagnostic = diagnose_series(series)
        assert diagnostic.ess is not None
        assert diagnostic.ess <= len(series)


def test_chain_summary_reports_spin_and_magnetization_levels() -> None:
    samples = np.random.default_rng(4).choice((-1, 1), size=(500, 3))

    summary = summarize_chain(samples, complete_sweeps_per_state=2)

    assert summary.recorded_states == 500
    assert summary.complete_sweeps_per_recorded_state == 2
    assert len(summary.spin_coordinates) == 3
    assert summary.minimum_spin_ess <= summary.median_spin_ess <= 500
    assert summary.magnetization.status == "estimated"
