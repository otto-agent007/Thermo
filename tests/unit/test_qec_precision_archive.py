"""Portable, read-only replay of the bounded QEC precision diagnostic archive."""

import hashlib
import json
import tarfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

ARCHIVE = (
    Path(__file__).resolve().parents[2]
    / "docs/research/qec-work-2026-11-14/archives/precision-diagnostic.tar.gz"
)


def test_bounded_qec_precision_archive_replays_exactly():
    """Independent primal enumeration checks the saved signed-Fourier enclosures."""
    manifest = json.loads(ARCHIVE.with_suffix(".json").read_text())
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == manifest["sha256"]
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        results = json.load(archive.extractfile("qec-precision-diagnostic/RESULTS.json"))
    assert results["complete"] is True
    rows = results["results"]
    assert len(rows) == 22
    mass_intervals = posterior_intervals = 0
    held = {name: Counter() for name in ("top1", "top4", "disagreement")}
    for row in rows:
        fixture = row["fixture"]
        m = fixture["m"]
        probabilities = [Fraction(*pair) for pair in fixture["p"]]
        assert len(probabilities) <= 10
        masses = [Fraction(0) for _ in range(2 ** (m + 1))]
        explanations = [[] for _ in masses]
        for pattern in range(2 ** len(probabilities)):
            event = 0
            weight = Fraction(1)
            for bit, (p, mask) in enumerate(zip(probabilities, fixture["masks"], strict=True)):
                occurs = (pattern >> bit) & 1
                weight *= p if occurs else 1 - p
                if occurs:
                    event ^= mask
            masses[event] += weight
            if weight:
                explanations[event].append(weight)
        assert sum(masses) == 1
        assert masses == [Fraction(*pair) for pair in row["truth_mass"]]
        for precision in row["precisions"]:
            for truth, interval in zip(masses, precision["mass"], strict=True):
                assert Fraction(interval[0]) <= truth <= Fraction(interval[1])
                mass_intervals += 1
            for syndrome, interval in enumerate(precision["posterior"]):
                total = masses[syndrome] + masses[syndrome + 2**m]
                if not total:
                    assert interval is None
                elif interval is not None:
                    truth = masses[syndrome] / total
                    assert Fraction(interval[0]) <= truth <= Fraction(interval[1])
                    posterior_intervals += 1
        for item in row["proxies"]:
            syndrome = item["syndrome"]
            total = masses[syndrome] + masses[syndrome + 2**m]
            posterior = masses[syndrome] / total
            truth_close = Fraction(9, 20) <= posterior <= Fraction(11, 20)
            assert item["truth_close"] == truth_close
            predictions = []
            for k in (1, 4):
                a = sum(sorted(explanations[syndrome], reverse=True)[:k], Fraction(0))
                b = sum(sorted(explanations[syndrome + 2**m], reverse=True)[:k], Fraction(0))
                margin = abs(a - b) / (a + b)
                prediction = None if a == b else int(a < b)
                proxy = item[f"top{k}"]
                assert Fraction(*proxy["margin"]) == margin
                assert proxy["prediction"] == prediction
                assert proxy["close"] == (margin <= Fraction(1, 10))
                predictions.append(prediction)
            disagreement = None not in predictions and predictions[0] != predictions[1]
            assert item["disagreement"] == disagreement
            if fixture["split"] == "heldout":
                for name in held:
                    selected = disagreement if name == "disagreement" else item[name]["close"]
                    held[name][(bool(selected), bool(truth_close))] += 1
    assert (mass_intervals, posterior_intervals) == (976, 450)
    assert held["top1"] == {(False, False): 44, (False, True): 2, (True, False): 2}
    assert held["top4"] == {(False, False): 45, (True, True): 2, (True, False): 1}
    assert held["disagreement"] == {(False, False): 46, (False, True): 1, (True, True): 1}
