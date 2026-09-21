"""Timing scopes distinguish nested construction from sampling without changing values."""


def test_nested_work_is_separate_and_disabled_outside_collection():
    from thermo_lab.runtime_work import collect_work, timed_work

    @timed_work("tables")
    def tables():
        return (1, 2)

    @timed_work("sampling")
    def sampler():
        return tables()

    with collect_work() as ledger:
        assert sampler() == (1, 2)
    snapshot = dict(ledger)
    sampler()
    assert ledger == snapshot
    assert ledger["sampling"]["calls"] == ledger["tables"]["calls"] == 1
    assert ledger["sampling"]["inclusive_seconds"] >= ledger["tables"]["inclusive_seconds"]
    assert ledger["sampling"]["exclusive_seconds"] >= 0


def test_failure_restores_collection_context():
    import pytest

    from thermo_lab.runtime_work import collect_work, timed_work

    @timed_work("failure")
    def fail():
        raise ValueError("test")

    with pytest.raises(ValueError), collect_work() as first:
        fail()
    with collect_work() as second:
        pass
    assert first["failure"]["calls"] == 1
    assert second == {}
