"""Codebook quantization of couplings and fields under a declared cap rule.

The codebook is M5b's rule (``meta_ebm_thermalization_core.round_parameters``)
with clipping and any width. On ``round_parameters``' own domain (bits in
{4, 8, 12}, every value within the cap) the layer calls ``round_parameters``
itself, so M5b agreement is reuse, not reimplementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from thermo_lab.constraints.model import PairwiseModel
from thermo_lab.constraints.request import ConstraintRequest
from thermo_lab.meta_ebm_thermalization_core import round_parameters

M5B_BITS = (4, 8, 12)


def codebook(x, cap: float, bits: int) -> np.ndarray:
    """2^bits - 1 symmetric levels on [-cap, cap]; values beyond the cap clip."""
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("cap must be positive and finite")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits < 2:
        raise ValueError("bits must be an integer >= 2")
    levels = 2 ** (bits - 1) - 1
    step = cap / levels
    return step * np.clip(np.rint(np.asarray(x, dtype=np.float64) / step), -levels, levels)


def _on_m5b_domain(x: np.ndarray, cap: float, bits: int) -> bool:
    return (
        bits in M5B_BITS
        and math.isfinite(cap)
        and cap > 0
        and x.ndim == 1
        and x.size > 0
        and bool(np.isfinite(x).all())
        and not bool(np.any(np.abs(x) > cap))
    )


def quantize_array(x, cap: float, bits: int) -> np.ndarray:
    """Quantize one array; delegates to ``round_parameters`` on its domain."""
    array = np.asarray(x, dtype=np.float64)
    if _on_m5b_domain(array, cap, bits):
        (rounded,), _ = round_parameters([array], cap, bits)
        return rounded
    return codebook(array, cap, bits)


def _summarize(error: np.ndarray, cap: float) -> dict:
    return {
        "max_absolute": float(error.max()),
        "mean_absolute": float(error.mean()),
        "max_cap_normalized": float(error.max() / cap),
        "mean_cap_normalized": float(error.mean() / cap),
    }


def rounding_summary(vectors: Sequence, cap: float, bits: int) -> dict:
    """M5b's ``rounding`` record (same fields, same reduction order), via the codebook."""
    errors = [np.abs(codebook(v, cap, bits) - np.asarray(v, dtype=np.float64)) for v in vectors]
    return {
        **_summarize(np.concatenate(errors), cap),
        "sites": [_summarize(e, cap) for e in errors],
    }


def cap_for(rule: str, couplings, fields, bits: int, cap: float | None = None):
    """(coupling cap, field cap) under one declared cap rule.

    ``absolute``: the declared ``cap`` for both. ``full``: max |coupling or field|
    for both. ``split``: max |coupling| and max |field| separately. ``coupling``:
    step = J, i.e. (2^(bits-1) - 1) * J for both, defined only when every nonzero
    coupling has the same magnitude J.
    """
    c = np.abs(np.asarray(couplings, dtype=np.float64)).ravel()
    f = np.abs(np.asarray(fields, dtype=np.float64)).ravel()
    coup = float(c.max(initial=0.0))
    field = float(f.max(initial=0.0))
    if rule == "absolute":
        if cap is None:
            raise ValueError("the absolute rule needs a cap")
        return float(cap), float(cap)
    if rule == "full":
        both = max(coup, field)
        return both, both
    if rule == "split":
        return coup, field
    if rule == "coupling":
        magnitudes = np.unique(c[c != 0])
        if magnitudes.size != 1:
            raise ValueError("the coupling cap rule needs one shared nonzero coupling magnitude")
        level = (2 ** (bits - 1) - 1) * float(magnitudes[0])
        return level, level
    raise ValueError(f"unknown cap rule {rule!r}")


def _caps(request: ConstraintRequest, couplings, fields) -> tuple[float, float]:
    if not request.applies("quantization"):
        raise ValueError("quantization is relaxed in this request")
    assert request.bits is not None
    return cap_for(request.cap_rule, couplings, fields, request.bits, request.cap)


def quantize(model: PairwiseModel, request: ConstraintRequest) -> tuple[PairwiseModel, dict]:
    """Quantize a model's weights and biases; returns the model and a rounding summary."""
    cap_c, cap_f = _caps(request, model.weights, model.bias)
    assert request.bits is not None
    weights = (
        quantize_array(model.weights, cap_c, request.bits) if len(model.weights) else model.weights
    )
    bias = quantize_array(model.bias, cap_f, request.bits)
    summary = {
        "cap_rule": request.cap_rule,
        "bits": request.bits,
        "coupling_cap": cap_c,
        "field_cap": cap_f,
        "couplings": rounding_summary([model.weights], cap_c, request.bits)
        if len(model.weights)
        else None,
        "fields": rounding_summary([model.bias], cap_f, request.bits),
    }
    return model.with_parameters(weights, bias), summary


def quantize_terms(
    terms: Mapping[str, np.ndarray],
    request: ConstraintRequest,
    coupling_keys: Sequence[str] = ("j_vv", "j_vh", "j_hh"),
    field_keys: Sequence[str] = ("b_v", "b_h"),
) -> dict:
    """Quantize a dict of spin terms (the associative-memory form) under one request."""
    couplings = np.concatenate([np.ravel(terms[k]) for k in coupling_keys])
    fields = np.concatenate([np.ravel(terms[k]) for k in field_keys])
    cap_c, cap_f = _caps(request, couplings, fields)
    assert request.bits is not None
    out = {k: quantize_array(terms[k], cap_c, request.bits) for k in coupling_keys}
    out.update({k: quantize_array(terms[k], cap_f, request.bits) for k in field_keys})
    return out
