"""Pure-function tests for the LLMDet labeler — no model, no torch, no network."""

import io

import pytest
from PIL import Image

from nounbox_labeler_llmdet import LLMDetLabeler
from nounbox_labeler_llmdet import labeler as mod
from nounbox_labeler_llmdet.labeler import Detection
from nounbox_sdk import BBox, Labeler

# the model's real BertTokenizerFast output for the prompt "carpet. sofa. chandelier."
# ['[CLS]', 'carpet', '.', 'sofa', '.', 'chan', '##del', '##ier', '.', '[SEP]']
IDS_THREE_CLASSES = [101, 10135, 1012, 10682, 1012, 9212, 9247, 3771, 1012, 102]
CLS_ID, SEP_ID, DOT_ID = 101, 102, 1012


def png(width, height, color="white"):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def detection(label, score, box):
    return Detection(label=label, score=score, box=box)


class FakeEngine:
    """Stub engine: remembers what it was called with and returns canned boxes."""

    def __init__(self, detections=(), model_id="fake"):
        self.detections = list(detections)
        self.model_id = model_id
        self.calls = []

    def detect(self, image, classes, score_threshold):
        self.calls.append((image.size, list(classes), score_threshold))
        return self.detections


# --- building the prompt ---


def test_build_prompt_joins_with_dots():
    assert mod.build_prompt(["carpet", "sofa", "chandelier"]) == (
        "carpet. sofa. chandelier."
    )


def test_build_prompt_lowercases_and_squeezes_spaces():
    """The processor strips case and extra spaces itself — we do it, predictably."""
    assert mod.build_prompt(["  Cutting   Board ", "SOFA"]) == "cutting board. sofa."


def test_build_prompt_single_class_still_ends_with_dot():
    assert mod.build_prompt(["carpet"]) == "carpet."


# --- the project classes from config ---


def test_resolve_classes_keeps_order_and_original_case():
    """Annotation.label has to match project_classes.name — we don't touch the case."""
    assert mod.resolve_classes({"classes": ["Carpet", "sofa", "Chandelier"]}) == [
        "Carpet",
        "sofa",
        "Chandelier",
    ]


def test_resolve_classes_strips_and_drops_empty():
    assert mod.resolve_classes({"classes": [" carpet ", "", "  ", "sofa"]}) == [
        "carpet",
        "sofa",
    ]


def test_resolve_classes_dedupes_case_insensitively():
    """A "Sofa" and a "sofa" would give two identical token spans in the prompt."""
    assert mod.resolve_classes({"classes": ["sofa", "Sofa", "SOFA"]}) == ["sofa"]


def test_resolve_classes_rejects_missing():
    with pytest.raises(ValueError, match="classes"):
        mod.resolve_classes({})


def test_resolve_classes_rejects_empty_list():
    with pytest.raises(ValueError, match="add the project classes"):
        mod.resolve_classes({"classes": []})


def test_resolve_classes_rejects_string():
    """The string "carpet" would iterate letter by letter — silent garbage."""
    with pytest.raises(ValueError, match="must be a list"):
        mod.resolve_classes({"classes": "carpet"})


def test_resolve_classes_rejects_dot_in_name():
    """A dot separates classes in the prompt: a class with a dot would split in two."""
    with pytest.raises(ValueError, match=r"contains '\.'"):
        mod.resolve_classes({"classes": ["u.s. flag"]})


# --- the 91-class limit ---


def test_class_limit_allows_exactly_max():
    classes = [f"class {i}" for i in range(mod.MAX_CLASSES)]
    assert mod.resolve_classes({"classes": classes}) == classes


def test_class_limit_rejects_one_over_with_both_numbers():
    classes = [f"class {i}" for i in range(mod.MAX_CLASSES + 1)]
    with pytest.raises(ValueError) as excinfo:
        mod.resolve_classes({"classes": classes})
    message = str(excinfo.value)
    assert str(mod.MAX_CLASSES) in message  # how many are allowed
    assert str(len(classes)) in message  # how many the project has
    assert "owlv2" in message  # where to go instead


def test_class_limit_counts_after_dedupe():
    """Duplicates must not push the project over the limit."""
    classes = [f"class {i}" for i in range(mod.MAX_CLASSES)] + ["CLASS 0", "class 1"]
    assert len(mod.resolve_classes({"classes": classes})) == mod.MAX_CLASSES


def test_prompt_token_limit_is_the_second_gate():
    """Checking for 91 classes is not enough when the names are long: the
    model's limit is in tokens, not in classes.
    """
    with pytest.raises(ValueError, match="tokens"):
        mod.check_prompt_tokens(319, 256, 91)
    assert mod.check_prompt_tokens(256, 256, 91) is None


# --- model selection ---


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "iSEE-Laboratory/llmdet_base"),
        ("", "iSEE-Laboratory/llmdet_base"),
        ("base", "iSEE-Laboratory/llmdet_base"),
        (" Tiny ", "iSEE-Laboratory/llmdet_tiny"),
        ("iSEE-Laboratory/llmdet_tiny", "iSEE-Laboratory/llmdet_tiny"),
    ],
)
def test_resolve_model(value, expected):
    assert mod.resolve_model({"model": value} if value is not None else {}) == expected


def test_resolve_model_rejects_unknown_alias():
    with pytest.raises(ValueError, match="unknown model"):
        mod.resolve_model({"model": "huge"})


# --- per-class token spans ---


def test_class_spans_splits_on_dots():
    spans = mod.class_spans(IDS_THREE_CLASSES, DOT_ID, {CLS_ID, SEP_ID}, 3)
    # 'chandelier' is split into three wordpieces — the class score averages them
    assert spans == [[1], [3], [5, 6, 7]]


def test_class_spans_without_trailing_dot():
    ids = [CLS_ID, 10135, DOT_ID, 10682, SEP_ID]
    assert mod.class_spans(ids, DOT_ID, {CLS_ID, SEP_ID}, 2) == [[1], [3]]


def test_class_spans_truncated_prompt_yields_fewer_spans():
    """A truncated prompt yields fewer class spans — the caller catches that."""
    ids = [CLS_ID, 10135, DOT_ID, SEP_ID]
    assert mod.class_spans(ids, DOT_ID, {CLS_ID, SEP_ID}, 3) == [[1]]


def test_class_spans_never_returns_more_than_requested():
    assert len(mod.class_spans(IDS_THREE_CLASSES, DOT_ID, {CLS_ID, SEP_ID}, 2)) == 2


# --- coordinates ---


def test_scale_box_converts_cxcywh_to_pixels():
    assert mod.scale_box(0.5, 0.5, 0.5, 0.25, 800, 400) == (200.0, 150.0, 600.0, 250.0)


def test_scale_box_clips_to_frame():
    """The model happily runs off-frame on cropped objects — so we clip."""
    assert mod.scale_box(0.5, 0.5, 2.0, 2.0, 100, 50) == (0.0, 0.0, 100.0, 50.0)


# --- NMS within a class ---


def test_nms_keeps_best_of_duplicates():
    best = detection("sofa", 0.9, (0, 0, 100, 100))
    duplicate = detection("sofa", 0.7, (5, 5, 105, 105))
    assert mod.nms_per_class([duplicate, best], 0.4) == [best]


def test_nms_keeps_overlapping_boxes_of_different_classes():
    """A "wine bottle" over a "bottle" is labeling, not a duplicate."""
    bottle = detection("bottle", 0.9, (0, 0, 100, 100))
    wine = detection("wine bottle", 0.8, (0, 0, 100, 100))
    assert set(mod.nms_per_class([bottle, wine], 0.4)) == {bottle, wine}


def test_nms_keeps_distinct_instances_of_same_class():
    left = detection("chair", 0.9, (0, 0, 100, 100))
    right = detection("chair", 0.8, (200, 0, 300, 100))
    assert set(mod.nms_per_class([left, right], 0.4)) == {left, right}


def test_nms_threshold_is_exclusive_on_equality():
    """IoU exactly at the threshold — no suppression."""
    a = detection("chair", 0.9, (0, 0, 100, 100))
    b = detection("chair", 0.8, (50, 0, 150, 100))  # IoU = 1/3
    assert len(mod.nms_per_class([a, b], 1 / 3)) == 2
    assert len(mod.nms_per_class([a, b], 0.3)) == 1


# --- conversion into Annotation ---


def test_to_annotations_builds_bbox_with_real_confidence():
    (ann,) = mod.to_annotations([detection("carpet", 0.87, (10, 20, 110, 220))])
    assert isinstance(ann.geometry, BBox)
    assert (ann.geometry.x, ann.geometry.y) == (10, 20)
    assert (ann.geometry.width, ann.geometry.height) == (100, 200)
    assert ann.label == "carpet"  # the project class name, not "text_line"
    assert ann.text is None  # detection, not OCR
    assert ann.confidence == 0.87  # the model's score, not a constant


def test_to_annotations_sorted_by_confidence():
    anns = mod.to_annotations(
        [
            detection("a", 0.4, (0, 0, 10, 10)),
            detection("b", 0.9, (20, 0, 30, 10)),
            detection("c", 0.6, (40, 0, 50, 10)),
        ]
    )
    assert [a.confidence for a in anns] == [0.9, 0.6, 0.4]


def test_score_threshold_filters_and_is_inclusive():
    detections = [
        detection("a", 0.34, (0, 0, 10, 10)),
        detection("b", 0.35, (20, 0, 30, 10)),
        detection("c", 0.99, (40, 0, 50, 10)),
    ]
    assert [a.label for a in mod.to_annotations(detections)] == ["c", "b"]


def test_default_score_threshold_is_the_measured_one():
    assert mod.DEFAULT_SCORE_THRESHOLD == 0.35
    assert mod.DEFAULT_NMS_IOU == 0.4


def test_score_threshold_zero_keeps_everything():
    detections = [detection("a", 0.01, (0, 0, 10, 10))]
    assert len(mod.to_annotations(detections, score_threshold=0.0)) == 1


def test_to_annotations_drops_degenerate_boxes():
    detections = [
        detection("a", 0.9, (10.0, 10.0, 10.2, 200.0)),  # a sliver 0.2 px wide
        detection("b", 0.9, (10.0, 10.0, 110.0, 210.0)),
    ]
    assert [a.label for a in mod.to_annotations(detections)] == ["b"]


def test_max_detections_caps_output():
    detections = [
        detection("a", 0.5 + i / 1000, (i * 20, 0, i * 20 + 10, 10)) for i in range(10)
    ]
    anns = mod.to_annotations(detections, max_detections=3)
    assert len(anns) == 3
    assert [a.confidence for a in anns] == [0.509, 0.508, 0.507]


def test_empty_detections_return_empty_list():
    assert mod.to_annotations([]) == []


# --- the image ---


def test_load_image_returns_rgb():
    image = mod.load_image(png(320, 200), 10_000_000)
    assert image.size == (320, 200)
    assert image.mode == "RGB"


def test_load_image_rejects_garbage():
    with pytest.raises(ValueError, match="cannot decode image"):
        mod.load_image(b"definitely not an image", 10_000_000)


def test_load_image_rejects_empty_payload():
    with pytest.raises(ValueError, match="empty image payload"):
        mod.load_image(b"", 10_000_000)


def test_load_image_rejects_oversized():
    with pytest.raises(ValueError, match="exceeds the limit"):
        mod.load_image(png(300, 300), 10_000)


def test_option_treats_null_as_default():
    """config comes from the job's JSON payload, where null turns up all too often."""
    assert mod.option({}, "score_threshold", 0.35) == 0.35
    assert mod.option({"score_threshold": None}, "score_threshold", 0.35) == 0.35
    assert mod.option({"score_threshold": 0}, "score_threshold", 0.35) == 0.0
    assert mod.option({"score_threshold": "0.5"}, "score_threshold", 0.35) == 0.5


def test_option_rejects_garbage():
    with pytest.raises(ValueError, match="must be a number"):
        mod.option({"nms_iou": "half"}, "nms_iou", 0.4)


def test_predict_rejects_nonpositive_max_detections(monkeypatch):
    monkeypatch.setattr(
        LLMDetLabeler, "_create_engine", staticmethod(lambda model_id: FakeEngine())
    )
    with pytest.raises(ValueError, match="max_detections must be positive"):
        LLMDetLabeler().predict(png(64, 64), {"classes": ["sofa"], "max_detections": 0})


def test_max_pixels_config_wins_over_env(monkeypatch):
    monkeypatch.setenv(mod.ENV_MAX_PIXELS, "123")
    assert mod.max_pixels({"max_pixels": 456}) == 456
    assert mod.max_pixels({}) == 123
    monkeypatch.delenv(mod.ENV_MAX_PIXELS)
    assert mod.max_pixels({}) == mod.DEFAULT_MAX_PIXELS


def test_max_pixels_rejects_nonpositive():
    with pytest.raises(ValueError, match="must be positive"):
        mod.max_pixels({"max_pixels": -1})


# --- the weight cache ---


def test_model_dir_from_env(monkeypatch, tmp_path):
    target = tmp_path / "weights"
    monkeypatch.setenv(mod.ENV_MODEL_DIR, str(target))
    assert mod.model_dir() == str(target)
    assert target.is_dir()


def test_model_dir_falls_back_to_hf_cache_when_unwritable(monkeypatch, tmp_path):
    (tmp_path / "file.txt").write_text("not a directory")
    monkeypatch.setenv(mod.ENV_MODEL_DIR, str(tmp_path / "file.txt" / "weights"))
    assert mod.model_dir() is None  # None = the stock huggingface cache


# --- lazy model loading ---


def test_engine_created_once_per_model(monkeypatch):
    created = []

    def fake_create(model_id):
        created.append(model_id)
        return FakeEngine(model_id=model_id)

    monkeypatch.setattr(LLMDetLabeler, "_create_engine", staticmethod(fake_create))
    instance = LLMDetLabeler()

    assert instance._engines == {}  # nothing is loaded before the first predict
    first = instance._engine("iSEE-Laboratory/llmdet_base")
    assert instance._engine("iSEE-Laboratory/llmdet_base") is first
    instance._engine("iSEE-Laboratory/llmdet_tiny")
    assert created == ["iSEE-Laboratory/llmdet_base", "iSEE-Laboratory/llmdet_tiny"]


def test_class_limit_checked_before_model_is_loaded(monkeypatch):
    def boom(model_id):
        raise AssertionError("model must not be loaded for an invalid config")

    monkeypatch.setattr(LLMDetLabeler, "_create_engine", staticmethod(boom))
    with pytest.raises(ValueError, match="owlv2"):
        LLMDetLabeler().predict(
            png(64, 64), {"classes": [f"class {i}" for i in range(200)]}
        )


def test_missing_classes_checked_before_model_is_loaded(monkeypatch):
    def boom(model_id):
        raise AssertionError("model must not be loaded for an invalid config")

    monkeypatch.setattr(LLMDetLabeler, "_create_engine", staticmethod(boom))
    with pytest.raises(ValueError, match="classes"):
        LLMDetLabeler().predict(png(64, 64), {})


def test_predict_wires_config_into_engine_and_postprocessing(monkeypatch):
    engine = FakeEngine(
        [
            detection("sofa", 0.90, (0, 0, 100, 100)),
            detection("sofa", 0.80, (4, 4, 104, 104)),  # duplicate -> NMS
            detection("carpet", 0.30, (0, 0, 50, 50)),  # below the threshold
        ]
    )
    monkeypatch.setattr(
        LLMDetLabeler, "_create_engine", staticmethod(lambda model_id: engine)
    )

    anns = LLMDetLabeler().predict(
        png(200, 200), {"classes": ["sofa", "carpet"], "model": "tiny"}
    )

    assert [(a.label, a.confidence) for a in anns] == [("sofa", 0.90)]
    (size, classes, threshold) = engine.calls[0]
    assert size == (200, 200)
    assert classes == ["sofa", "carpet"]
    assert threshold == mod.DEFAULT_SCORE_THRESHOLD


def test_predict_honours_explicit_thresholds(monkeypatch):
    engine = FakeEngine(
        [
            detection("sofa", 0.90, (0, 0, 100, 100)),
            detection("sofa", 0.80, (4, 4, 104, 104)),
            detection("carpet", 0.30, (0, 0, 50, 50)),
        ]
    )
    monkeypatch.setattr(
        LLMDetLabeler, "_create_engine", staticmethod(lambda model_id: engine)
    )

    anns = LLMDetLabeler().predict(
        png(200, 200),
        {"classes": ["sofa", "carpet"], "score_threshold": 0.25, "nms_iou": 0.99},
    )

    assert [(a.label, a.confidence) for a in anns] == [
        ("sofa", 0.90),
        ("sofa", 0.80),
        ("carpet", 0.30),
    ]
    assert engine.calls[0][2] == 0.25  # threshold goes to the engine too — no waste


# --- the platform contract ---


def test_satisfies_sdk_protocol():
    instance = LLMDetLabeler()
    assert isinstance(instance, Labeler)
    assert instance.name == "llmdet"
    assert {c.value for c in instance.capabilities} == {"detection"}
