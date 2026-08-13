"""Тесты чистых функций LLMDet-лейблера — без модели, без torch и без сети."""

import io

import pytest
from PIL import Image

from autolabel_labeler_llmdet import LLMDetLabeler
from autolabel_labeler_llmdet import labeler as mod
from autolabel_labeler_llmdet.labeler import Detection
from autolabelui_sdk import BBox, Labeler

# реальный выход BertTokenizerFast модели на промпт "carpet. sofa. chandelier."
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
    """Движок-заглушка: помнит, с чем его позвали, и отдаёт готовые рамки."""

    def __init__(self, detections=(), model_id="fake"):
        self.detections = list(detections)
        self.model_id = model_id
        self.calls = []

    def detect(self, image, classes, score_threshold):
        self.calls.append((image.size, list(classes), score_threshold))
        return self.detections


# --- сборка промпта ---


def test_build_prompt_joins_with_dots():
    assert mod.build_prompt(["carpet", "sofa", "chandelier"]) == (
        "carpet. sofa. chandelier."
    )


def test_build_prompt_lowercases_and_squeezes_spaces():
    """Регистр и лишние пробелы режет сам processor — делаем это сами и предсказуемо."""
    assert mod.build_prompt(["  Cutting   Board ", "SOFA"]) == "cutting board. sofa."


def test_build_prompt_single_class_still_ends_with_dot():
    assert mod.build_prompt(["carpet"]) == "carpet."


# --- классы проекта из config ---


def test_resolve_classes_keeps_order_and_original_case():
    """label в Annotation должен совпасть с project_classes.name — регистр не трогаем."""
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
    """«Sofa» и «sofa» дали бы два одинаковых диапазона токенов в промпте."""
    assert mod.resolve_classes({"classes": ["sofa", "Sofa", "SOFA"]}) == ["sofa"]


def test_resolve_classes_rejects_missing():
    with pytest.raises(ValueError, match="classes"):
        mod.resolve_classes({})


def test_resolve_classes_rejects_empty_list():
    with pytest.raises(ValueError, match="добавьте классы проекта"):
        mod.resolve_classes({"classes": []})


def test_resolve_classes_rejects_string():
    """Строка «carpet» проитерировалась бы по буквам — это молчаливый мусор."""
    with pytest.raises(ValueError, match="must be a list"):
        mod.resolve_classes({"classes": "carpet"})


def test_resolve_classes_rejects_dot_in_name():
    """Точка — разделитель классов в промпте: класс с точкой разъехался бы на два."""
    with pytest.raises(ValueError, match=r"contains '\.'"):
        mod.resolve_classes({"classes": ["u.s. flag"]})


# --- лимит 91 класса ---


def test_class_limit_allows_exactly_max():
    classes = [f"class {i}" for i in range(mod.MAX_CLASSES)]
    assert mod.resolve_classes({"classes": classes}) == classes


def test_class_limit_rejects_one_over_with_both_numbers():
    classes = [f"class {i}" for i in range(mod.MAX_CLASSES + 1)]
    with pytest.raises(ValueError) as excinfo:
        mod.resolve_classes({"classes": classes})
    message = str(excinfo.value)
    assert str(mod.MAX_CLASSES) in message  # сколько можно
    assert str(len(classes)) in message  # сколько у проекта
    assert "owlv2" in message  # куда идти


def test_class_limit_counts_after_dedupe():
    """Дубли не должны выталкивать проект за лимит."""
    classes = [f"class {i}" for i in range(mod.MAX_CLASSES)] + ["CLASS 0", "class 1"]
    assert len(mod.resolve_classes({"classes": classes})) == mod.MAX_CLASSES


def test_prompt_token_limit_is_the_second_gate():
    """91 класса мало, если имена длинные: лимит модели — в токенах, не в классах."""
    with pytest.raises(ValueError, match="токенов"):
        mod.check_prompt_tokens(319, 256, 91)
    assert mod.check_prompt_tokens(256, 256, 91) is None


# --- выбор модели ---


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


# --- диапазоны токенов по классам ---


def test_class_spans_splits_on_dots():
    spans = mod.class_spans(IDS_THREE_CLASSES, DOT_ID, {CLS_ID, SEP_ID}, 3)
    # 'chandelier' разбит на три wordpiece'а — скор класса усредняется по ним
    assert spans == [[1], [3], [5, 6, 7]]


def test_class_spans_without_trailing_dot():
    ids = [CLS_ID, 10135, DOT_ID, 10682, SEP_ID]
    assert mod.class_spans(ids, DOT_ID, {CLS_ID, SEP_ID}, 2) == [[1], [3]]


def test_class_spans_truncated_prompt_yields_fewer_spans():
    """Если промпт обрезан, классов-диапазонов меньше — вызывающий это ловит."""
    ids = [CLS_ID, 10135, DOT_ID, SEP_ID]
    assert mod.class_spans(ids, DOT_ID, {CLS_ID, SEP_ID}, 3) == [[1]]


def test_class_spans_never_returns_more_than_requested():
    assert len(mod.class_spans(IDS_THREE_CLASSES, DOT_ID, {CLS_ID, SEP_ID}, 2)) == 2


# --- координаты ---


def test_scale_box_converts_cxcywh_to_pixels():
    assert mod.scale_box(0.5, 0.5, 0.5, 0.25, 800, 400) == (200.0, 150.0, 600.0, 250.0)


def test_scale_box_clips_to_frame():
    """Модель уверенно вылезает за кадр на срезанных объектах — обрезаем."""
    assert mod.scale_box(0.5, 0.5, 2.0, 2.0, 100, 50) == (0.0, 0.0, 100.0, 50.0)


# --- NMS внутри класса ---


def test_nms_keeps_best_of_duplicates():
    best = detection("sofa", 0.9, (0, 0, 100, 100))
    duplicate = detection("sofa", 0.7, (5, 5, 105, 105))
    assert mod.nms_per_class([duplicate, best], 0.4) == [best]


def test_nms_keeps_overlapping_boxes_of_different_classes():
    """«wine bottle» поверх «bottle» — это разметка, а не дубль."""
    bottle = detection("bottle", 0.9, (0, 0, 100, 100))
    wine = detection("wine bottle", 0.8, (0, 0, 100, 100))
    assert set(mod.nms_per_class([bottle, wine], 0.4)) == {bottle, wine}


def test_nms_keeps_distinct_instances_of_same_class():
    left = detection("chair", 0.9, (0, 0, 100, 100))
    right = detection("chair", 0.8, (200, 0, 300, 100))
    assert set(mod.nms_per_class([left, right], 0.4)) == {left, right}


def test_nms_threshold_is_exclusive_on_equality():
    """IoU ровно на пороге — не подавляем."""
    a = detection("chair", 0.9, (0, 0, 100, 100))
    b = detection("chair", 0.8, (50, 0, 150, 100))  # IoU = 1/3
    assert len(mod.nms_per_class([a, b], 1 / 3)) == 2
    assert len(mod.nms_per_class([a, b], 0.3)) == 1


# --- разбор в Annotation ---


def test_to_annotations_builds_bbox_with_real_confidence():
    (ann,) = mod.to_annotations([detection("carpet", 0.87, (10, 20, 110, 220))])
    assert isinstance(ann.geometry, BBox)
    assert (ann.geometry.x, ann.geometry.y) == (10, 20)
    assert (ann.geometry.width, ann.geometry.height) == (100, 200)
    assert ann.label == "carpet"  # имя класса проекта, а не "text_line"
    assert ann.text is None  # detection, не OCR
    assert ann.confidence == 0.87  # скор модели, не константа


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
        detection("a", 0.9, (10.0, 10.0, 10.2, 200.0)),  # полоска шириной 0.2 px
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


# --- картинка ---


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
    """config приезжает из JSON-payload задачи: null там встречается чаще, чем нужно."""
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


# --- кеш весов ---


def test_model_dir_from_env(monkeypatch, tmp_path):
    target = tmp_path / "weights"
    monkeypatch.setenv(mod.ENV_MODEL_DIR, str(target))
    assert mod.model_dir() == str(target)
    assert target.is_dir()


def test_model_dir_falls_back_to_hf_cache_when_unwritable(monkeypatch, tmp_path):
    (tmp_path / "file.txt").write_text("not a directory")
    monkeypatch.setenv(mod.ENV_MODEL_DIR, str(tmp_path / "file.txt" / "weights"))
    assert mod.model_dir() is None  # None = штатный кеш huggingface


# --- ленивая загрузка модели ---


def test_engine_created_once_per_model(monkeypatch):
    created = []

    def fake_create(model_id):
        created.append(model_id)
        return FakeEngine(model_id=model_id)

    monkeypatch.setattr(LLMDetLabeler, "_create_engine", staticmethod(fake_create))
    instance = LLMDetLabeler()

    assert instance._engines == {}  # ничего не грузится до первого predict
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
            detection("sofa", 0.80, (4, 4, 104, 104)),  # дубль -> NMS
            detection("carpet", 0.30, (0, 0, 50, 50)),  # ниже порога
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
    assert engine.calls[0][2] == 0.25  # порог уходит и в движок — лишнего не считаем


# --- контракт платформы ---


def test_satisfies_sdk_protocol():
    instance = LLMDetLabeler()
    assert isinstance(instance, Labeler)
    assert instance.name == "llmdet"
    assert {c.value for c in instance.capabilities} == {"detection"}
