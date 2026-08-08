"""Conservative within-chain autocorrelation and effective-sample diagnostics.

The estimator uses Geyer's initial-positive sequence of adjacent
autocorrelation pairs. Pair sums are truncated at the first non-positive pair
and monotonized by replacing increases with the preceding smaller value.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from thermo_lab.records import FrozenModel

MIN_DIAGNOSTIC_STATES = 4


class SeriesDiagnostic(FrozenModel):
    recorded_states: int
    status: Literal["estimated", "constant_series", "insufficient_length"]
    lag_1_autocorrelation: float | None
    integrated_autocorrelation_time: float | None
    ess: float | None
    estimator: str = "Geyer initial-positive sequence with monotone paired sums"


class ChainDiagnostics(FrozenModel):
    recorded_states: int
    complete_sweeps_per_recorded_state: int
    spin_coordinates: tuple[SeriesDiagnostic, ...]
    minimum_spin_ess: float | None
    median_spin_ess: float | None
    magnetization: SeriesDiagnostic


def diagnose_series(values: ArrayLike) -> SeriesDiagnostic:
    """Estimate scalar autocorrelation time and ESS without overstating short traces."""

    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError("Autocorrelation diagnostics require a one-dimensional scalar series")
    n_states = int(series.size)
    if n_states < MIN_DIAGNOSTIC_STATES:
        return SeriesDiagnostic(
            recorded_states=n_states,
            status="insufficient_length",
            lag_1_autocorrelation=None,
            integrated_autocorrelation_time=None,
            ess=None,
        )
    if not np.all(np.isfinite(series)):
        raise ValueError("Autocorrelation diagnostics require finite values")

    centered = series - series.mean()
    variance = float(np.dot(centered, centered) / n_states)
    if variance <= np.finfo(np.float64).eps:
        return SeriesDiagnostic(
            recorded_states=n_states,
            status="constant_series",
            lag_1_autocorrelation=None,
            integrated_autocorrelation_time=None,
            ess=0.0,
        )

    fft_size = 1 << (2 * n_states - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    autocovariances = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:n_states]
    autocorrelations = np.clip(autocovariances / (n_states * variance), -1.0, 1.0)

    paired_sums: list[float] = []
    for first_lag in range(0, n_states - 1, 2):
        pair_sum = autocorrelations[first_lag] + autocorrelations[first_lag + 1]
        if pair_sum <= 0:
            break
        if paired_sums:
            pair_sum = min(float(pair_sum), paired_sums[-1])
        paired_sums.append(float(pair_sum))

    tau = float(np.clip(-1.0 + 2.0 * sum(paired_sums), 1.0, float(n_states)))
    ess = float(np.clip(n_states / tau, 0.0, float(n_states)))
    return SeriesDiagnostic(
        recorded_states=n_states,
        status="estimated",
        lag_1_autocorrelation=float(autocorrelations[1]),
        integrated_autocorrelation_time=tau,
        ess=ess,
    )


def summarize_chain(samples: ArrayLike, *, complete_sweeps_per_state: int) -> ChainDiagnostics:
    """Summarize coordinates and magnetization without flattening correlated states."""

    chain = np.asarray(samples)
    if chain.ndim != 2 or chain.shape[1] < 1:
        raise ValueError("Chain diagnostics require shape (recorded_states, coordinates)")
    if complete_sweeps_per_state < 1:
        raise ValueError("complete_sweeps_per_state must be at least one")
    coordinates = tuple(diagnose_series(chain[:, index]) for index in range(chain.shape[1]))
    available_ess = [item.ess for item in coordinates if item.ess is not None]
    return ChainDiagnostics(
        recorded_states=int(chain.shape[0]),
        complete_sweeps_per_recorded_state=complete_sweeps_per_state,
        spin_coordinates=coordinates,
        minimum_spin_ess=min(available_ess) if available_ess else None,
        median_spin_ess=float(np.median(available_ess)) if available_ess else None,
        magnetization=diagnose_series(chain.mean(axis=1)),
    )
