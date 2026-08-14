"""The HTTP labeler and its two platform-supplied subclasses.

Nothing here touches the network: httpx.post is replaced with a recorder.
"""

import json

import httpx
import pytest

from nounbox_labeler_http import (
    HttpLabeler,
    ModalGpuDetectLabeler,
    ModalGpuLabeler,
)
from nounbox_labeler_http import labeler as labeler_module


class Recorder:
    """Stands in for httpx.post: remembers the call, answers with a canned body."""

    def __init__(self, payload=None, status_code=200, text=""):
        self.payload = payload if payload is not None else {"annotations": []}
        self.status_code = status_code
        self.text = text
        self.calls: list[dict] = []

    def __call__(self, url, content=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "content": content, "headers": headers, "timeout": timeout}
        )
        return self

    # the response side
    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=None
            )


@pytest.fixture
def post(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(labeler_module.httpx, "post", recorder)
    return recorder


def sent_config(recorder: Recorder) -> dict:
    return json.loads(recorder.calls[-1]["headers"]["X-Labeler-Config"])


BOX = {
    "annotations": [
        {
            "geometry": {"type": "bbox", "x": 1, "y": 2, "width": 3, "height": 4},
            "label": "sofa",
            "confidence": 0.9,
        }
    ]
}


# --- the generic engine: its documented contract must not move ---


def test_http_forwards_only_backend_config(post):
    HttpLabeler().predict(
        b"img",
        {
            "endpoint": "https://example.test/predict",
            "backend_config": {"lang": "ru"},
            # a third-party endpoint has never seen these and must not start now
            "classes": ["sofa"],
            "score_threshold": 0.25,
        },
    )

    assert sent_config(post) == {"lang": "ru"}


def test_http_without_backend_config_sends_an_empty_object(post):
    HttpLabeler().predict(b"img", {"endpoint": "https://example.test/predict"})

    assert sent_config(post) == {}


def test_endpoint_is_required():
    with pytest.raises(ValueError, match="endpoint not set"):
        HttpLabeler().predict(b"img", {})


def test_api_key_becomes_a_bearer_header(post):
    HttpLabeler().predict(
        b"img", {"endpoint": "https://example.test/predict", "api_key": "s3cret"}
    )

    assert post.calls[-1]["headers"]["Authorization"] == "Bearer s3cret"


# --- 4xx versus 5xx ---


def test_4xx_is_a_value_error(monkeypatch):
    """run_autolabel only stops the job on ValueError; anything else counts one
    failed image and the run ends DONE with nothing labeled."""
    monkeypatch.setattr(
        labeler_module.httpx,
        "post",
        Recorder(status_code=401, text="Unauthorized"),
    )

    with pytest.raises(ValueError, match="401"):
        HttpLabeler().predict(b"img", {"endpoint": "https://example.test/predict"})


def test_5xx_stays_an_http_error(monkeypatch):
    # a 500 really can be one flaky image — the job should survive it
    monkeypatch.setattr(
        labeler_module.httpx, "post", Recorder(status_code=503, text="oops")
    )

    with pytest.raises(httpx.HTTPStatusError):
        HttpLabeler().predict(b"img", {"endpoint": "https://example.test/predict"})


def test_oversized_config_is_refused_before_the_request(post):
    huge = ["class-name-%04d" % i for i in range(2000)]

    with pytest.raises(ValueError, match="too large"):
        ModalGpuDetectLabeler().predict(
            b"img", {"endpoint": "https://example.test/predict", "classes": huge}
        )

    assert post.calls == []  # an unreadable 431 never happens


# --- parsing ---


def test_missing_text_becomes_none(monkeypatch):
    monkeypatch.setattr(labeler_module.httpx, "post", Recorder(BOX))

    annotations = HttpLabeler().predict(
        b"img", {"endpoint": "https://example.test/predict"}
    )

    assert len(annotations) == 1
    assert annotations[0].text is None  # exactly what the CPU detector stores
    assert annotations[0].label == "sofa"
    assert annotations[0].confidence == pytest.approx(0.9)


# --- modal_gpu (OCR) ---


def test_modal_gpu_forwards_its_recipe_keys(post):
    ModalGpuLabeler().predict(
        b"img",
        {
            "endpoint": "https://example.test/predict",
            "api_key": "s3cret",
            "timeout": 60,
            "lang": "ru",
            "text_rec_score_thresh": 0.3,
        },
    )

    # lang was silently dropped before this: the header payload was always {}
    assert sent_config(post) == {"lang": "ru", "text_rec_score_thresh": 0.3}


def test_modal_gpu_is_recognition_only():
    from nounbox_sdk import Capability

    assert ModalGpuLabeler.capabilities == {Capability.RECOGNITION}


# --- modal_gpu_detect ---


def test_detect_forwards_classes_and_thresholds(post):
    ModalGpuDetectLabeler().predict(
        b"img",
        {
            "endpoint": "https://example.test/predict",
            "classes": ["sofa"],
            "score_threshold": 0.3,
            "nms_iou": 0.5,
            "model": "google/owlv2-base-patch16-ensemble",
        },
    )

    assert sent_config(post) == {
        "classes": ["sofa"],
        "score_threshold": 0.3,
        "nms_iou": 0.5,
        "model": "google/owlv2-base-patch16-ensemble",
    }


def test_ours_never_leak_into_the_header(post):
    ModalGpuDetectLabeler().predict(
        b"img",
        {
            "endpoint": "https://example.test/predict",
            "api_key": "s3cret",
            "timeout": 42,
            "classes": ["sofa"],
        },
    )

    payload = sent_config(post)
    assert "api_key" not in payload
    assert "endpoint" not in payload
    assert "timeout" not in payload
    assert "s3cret" not in json.dumps(payload)


def test_explicit_backend_config_wins(post):
    ModalGpuDetectLabeler().predict(
        b"img",
        {
            "endpoint": "https://example.test/predict",
            "classes": ["sofa"],
            "score_threshold": 0.25,
            "backend_config": {"score_threshold": 0.9},
        },
    )

    assert sent_config(post)["score_threshold"] == 0.9


def test_detect_without_classes_is_a_value_error(post):
    with pytest.raises(ValueError, match="no classes given"):
        ModalGpuDetectLabeler().predict(
            b"img", {"endpoint": "https://example.test/predict", "classes": []}
        )

    assert post.calls == []


def test_label_outside_the_class_list_is_refused(monkeypatch):
    """A stale OCR app under this URL answers with text_line boxes. Dropping
    them quietly is the failure being fixed: export routes unknown labels into
    skipped_labels and the user ends up with an empty dataset."""
    monkeypatch.setattr(
        labeler_module.httpx,
        "post",
        Recorder(
            {
                "annotations": [
                    {
                        "geometry": {
                            "type": "bbox",
                            "x": 1,
                            "y": 2,
                            "width": 3,
                            "height": 4,
                        },
                        "label": "text_line",
                    }
                ]
            }
        ),
    )

    with pytest.raises(ValueError, match="text_line"):
        ModalGpuDetectLabeler().predict(
            b"img", {"endpoint": "https://example.test/predict", "classes": ["sofa"]}
        )


def test_matching_labels_pass_through(monkeypatch):
    monkeypatch.setattr(labeler_module.httpx, "post", Recorder(BOX))

    annotations = ModalGpuDetectLabeler().predict(
        b"img", {"endpoint": "https://example.test/predict", "classes": ["sofa"]}
    )

    assert [a.label for a in annotations] == ["sofa"]


def test_detect_is_detection_only():
    from nounbox_sdk import Capability

    assert ModalGpuDetectLabeler.capabilities == {Capability.DETECTION}
    assert ModalGpuDetectLabeler.name == "modal_gpu_detect"
