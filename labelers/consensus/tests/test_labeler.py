"""ConsensusLabeler configuration tests (no engines are actually run)."""

import pytest

from nounbox_labeler_consensus import ConsensusLabeler


def test_unknown_engine_error_lists_available():
    with pytest.raises(ValueError, match=r"'nope' is not installed; available: "):
        ConsensusLabeler().predict(b"", {"engines": [{"name": "nope"}]})


def test_engine_entry_without_name():
    with pytest.raises(ValueError, match='needs a "name"'):
        ConsensusLabeler().predict(b"", {"engines": [{"config": {}}]})


def test_cannot_nest_itself():
    with pytest.raises(ValueError, match="cannot include itself"):
        ConsensusLabeler().predict(b"", {"engines": [{"name": "consensus"}]})


from nounbox_sdk import Annotation, BBox


class _Stub:
    def __init__(self, name, annotations):
        self.name = name
        self.version = "0"
        self.capabilities = set()
        self.annotations = annotations
        self.calls = []

    def predict(self, image, config):
        self.calls.append((image, config))
        return list(self.annotations)


def test_predict_dispatch_primary_and_per_engine_config(monkeypatch):
    import nounbox_labeler_consensus.labeler as mod

    first = _Stub(
        "eng_a",
        [Annotation(geometry=BBox(x=0, y=0, width=100, height=20), text="primary")],
    )
    second = _Stub(
        "eng_b",
        [Annotation(geometry=BBox(x=1, y=1, width=99, height=19), text="primary")],
    )
    monkeypatch.setattr(mod, "load_labelers", lambda: {"eng_a": first, "eng_b": second})

    result = ConsensusLabeler().predict(
        b"img",
        {"engines": [{"name": "eng_a"}, {"name": "eng_b", "config": {"k": 1}}]},
    )

    assert first.calls[0][0] == b"img" and not first.calls[0][1]
    assert second.calls[0][1] == {"k": 1}
    assert len(result) == 1
    assert result[0].text == "primary"
    assert result[0].attrs["consensus"]["engines"] == ["eng_a", "eng_b"]


def test_predict_duplicate_engine_runs_twice(monkeypatch):
    import nounbox_labeler_consensus.labeler as mod

    stub = _Stub("eng_a", [])
    monkeypatch.setattr(mod, "load_labelers", lambda: {"eng_a": stub})

    assert ConsensusLabeler().predict(b"x", {"engines": ["eng_a", "eng_a"]}) == []
    assert len(stub.calls) == 2


def test_predict_empty_engines_list_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        ConsensusLabeler().predict(b"", {"engines": []})
