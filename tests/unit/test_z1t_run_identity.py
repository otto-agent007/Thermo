"""Small filesystem-only tests; never load Z1T or run training."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "z1t_runner", Path(__file__).parents[2] / "docs/research/z1t_noise_finetune.py"
)
ft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ft)


def test_changed_record_with_same_weights_is_retained(tmp_path):
    a = {"w": np.array([1.0])}
    ft.archive_step(str(tmp_path), 1, a, {"eval": 1})
    ft.archive_step(str(tmp_path), 1, a, {"eval": 2})
    assert json.loads((tmp_path / "step_000001_v2/record.json").read_text()) == {"eval": 2}


def test_json_corruption_does_not_replace_previous(tmp_path, monkeypatch):
    p = tmp_path / "record.json"
    ft.write_json(str(p), {"a": 1})
    original = ft.json.dump
    monkeypatch.setattr(
        ft.json, "dump", lambda obj, stream, **kw: original({"wrong": 3}, stream, **kw)
    )
    with pytest.raises(RuntimeError, match="SAVE FAILED"):
        ft.write_json(str(p), {"a": 2})
    assert json.loads(p.read_text()) == {"a": 1}


def test_run_identity_resume_and_rejection(tmp_path):
    out = tmp_path / "run"
    identity = {"inputs": {"data": "abc"}, "settings": {"N": 32}}
    prov = {"utc": "first", "versions": {"numpy": "1"}, "dtype": "float32"}
    with ft.run_session(str(out), identity, prov) as first:
        assert first["run_id"]
        with pytest.raises(RuntimeError, match="writer"):
            with ft.run_session(str(out), identity, prov):
                pass
    original = (out / "provenance.json").read_bytes()
    with ft.run_session(str(out), identity, dict(prov, utc="second")) as second:
        assert first["run_id"] == second["run_id"]
    assert (out / "provenance.json").read_bytes() == original
    assert len(list((out / "invocations").glob("*.json"))) == 2
    for changed in [
        {"inputs": {"data": "changed"}, "settings": {"N": 32}},
        {"inputs": {"data": "abc"}, "settings": {"N": 64}},
    ]:
        with pytest.raises(RuntimeError, match="identity"):
            with ft.run_session(str(out), changed, prov):
                pass
    assert len(list((out / "invocations").glob("*.json"))) == 2


def test_legacy_output_is_not_adopted(tmp_path):
    (tmp_path / "start.json").write_text("{}")
    with pytest.raises(RuntimeError, match="legacy"):
        with ft.run_session(str(tmp_path), {}, {}):
            pass
    assert not (tmp_path / "run-manifest.json").exists()


def test_identity_tracks_inputs_and_dirty_upstream(tmp_path):
    repo = tmp_path / "repo"
    (repo / "research").mkdir(parents=True)
    code = repo / "research/model.py"
    code.write_text("a=1")
    data = tmp_path / "data"
    data.write_text("a")
    args = SimpleNamespace(
        repo=str(repo),
        weights=str(data),
        out=str(tmp_path / "out"),
        arms="noisy",
        debug_stop_at=0,
        N=32,
    )
    a = ft.scientific_identity(args, __file__)
    args.arms = "control"
    assert ft.scientific_identity(args, __file__) == a
    data.write_text("b")
    assert ft.scientific_identity(args, __file__) != a
    data.write_text("a")
    code.write_text("a=2")
    assert ft.scientific_identity(args, __file__) != a


def test_pairing_rejects_same_length_different_windows():
    with pytest.raises(RuntimeError, match="identity"):
        ft.require_identity({"sig": {"window_id": "a"}}, {"window_id": "b"}, "final")


def test_corrupt_json_is_not_a_missing_cache(tmp_path):
    p = tmp_path / "start.json"
    p.write_text("{")
    with pytest.raises(RuntimeError, match="corrupt"):
        ft.read_json(str(p))


def test_interruption_releases_lock_and_preserves_manifest(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with ft.run_session(str(tmp_path), {"N": 32}, {}):
            raise KeyboardInterrupt()
    with ft.run_session(str(tmp_path), {"N": 32}, {}) as resumed:
        assert resumed["run_id"]


def test_npz_corruption_does_not_replace_previous(tmp_path, monkeypatch):
    p = tmp_path / "weights.npz"
    ft.save_npz_verified(str(p), {"w": np.array([1.0])})
    before = p.read_bytes()

    def truncated(f, **arrays):
        f.write(b"broken zip")

    monkeypatch.setattr(ft.np, "savez", truncated)
    with pytest.raises(RuntimeError, match="SAVE FAILED"):
        ft.save_npz_verified(str(p), {"w": np.array([2.0])})
    assert p.read_bytes() == before


def test_snapshot_corruption_is_rejected(tmp_path):
    a = {"w": np.array([1.0])}
    ft.archive_step(str(tmp_path), 1, a, {"x": 1})
    p = tmp_path / "step_000001/body_params.npz"
    p.chmod(0o644)
    p.write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="fails its manifest"):
        ft.archive_step(str(tmp_path), 1, a, {"x": 1})


def test_final_checkpoint_recovers_result_after_publication_interruption():
    sig = {"run_id": "x", "windows": {"owt": "tokens"}}
    result = {"sig": sig, "arm": "noisy", "row": {"step": 2}, "per": {"owt": [1, 2]}}
    meta = {"sig": sig, "arm": "noisy", "step": 2, "final_result": result}
    assert ft.checkpoint_final(meta, sig, "noisy", 2) == result
    with pytest.raises(RuntimeError, match="arm"):
        ft.checkpoint_final(meta, sig, "control", 2)
    with pytest.raises(RuntimeError, match="final result"):
        ft.checkpoint_final(dict(meta, final_result=None), sig, "noisy", 2)


@pytest.mark.parametrize("content", ["{}", "null", "[]", '{"x":NaN}', '{"x":Infinity}'])
def test_invalid_cache_json_fails_closed(tmp_path, content):
    p = tmp_path / "cache.json"
    p.write_text(content)
    with pytest.raises(RuntimeError, match="corrupt"):
        ft.read_json(str(p))


def test_completion_publication_interruption_is_retryable(tmp_path, monkeypatch):
    sig = {"run_id": "r"}
    result = {"sig": sig, "arm": "noisy", "row": {"step": 2}, "per": {"owt": [1, 2]}}
    meta = {
        "sig": sig,
        "arm": "noisy",
        "step": 2,
        "log": [{"step": 0}, {"step": 2}],
        "final_result": result,
    }
    original = ft.write_json

    def interrupted(path, obj):
        if path.endswith("final.json"):
            raise OSError("simulated process interruption before final publication")
        return original(path, obj)

    monkeypatch.setattr(ft, "write_json", interrupted)
    with pytest.raises(OSError):
        ft.recover_completed(str(tmp_path), meta, sig, "noisy", 2, {}, "train_exact")
    assert json.loads((tmp_path / "log.json").read_text())["log"][-1]["step"] == 2
    monkeypatch.setattr(ft, "write_json", original)
    assert ft.recover_completed(str(tmp_path), meta, sig, "noisy", 2, {}, "train_exact") == result
    assert json.loads((tmp_path / "final.json").read_text()) == result
