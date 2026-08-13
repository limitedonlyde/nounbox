"""Pure-function tests for the RapidOCR labeler — no model, no network."""

import io

import pytest
from PIL import Image

from nounbox_labeler_rapidocr import RapidOCRLabeler
from nounbox_labeler_rapidocr import labeler as mod
from nounbox_sdk import Labeler

# what actually comes out of RapidOCROutput: TL, TR, BR, BL
BOX_A = [[86.0, 101.0], [776.0, 102.0], [776.0, 135.0], [86.0, 134.0]]
BOX_B = [[86.0, 160.0], [500.0, 160.0], [500.0, 190.0], [86.0, 190.0]]
SKEWED = [[172.0, 99.0], [868.0, 135.0], [866.0, 181.0], [170.0, 145.0]]


def shoelace(points):
    """Signed area: >0 = clockwise in screen coordinates (y pointing down)."""
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        total += x1 * y2 - x2 * y1
    return total / 2


def png(width, height, color="white"):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


# --- converting RapidOCR output into Annotation ---


def test_converts_lines_to_annotations():
    anns = mod.to_annotations(
        [BOX_A, BOX_B],
        ("АКТ ВЫПОЛНЕННЫХ РАБОТ № 147/2026", "Исполнитель"),
        (0.98514, 0.9429),
    )
    assert [a.label for a in anns] == ["text_line", "text_line"]
    assert [a.text for a in anns] == ["АКТ ВЫПОЛНЕННЫХ РАБОТ № 147/2026", "Исполнитель"]
    # confidence is REAL: per line, exactly as the model gave it, not a constant
    assert [a.confidence for a in anns] == [0.98514, 0.9429]
    assert all(isinstance(a.confidence, float) for a in anns)


def test_polygon_is_four_points_clockwise():
    (ann,) = mod.to_annotations([BOX_A], ("line",), (0.99,))
    assert ann.geometry == [(86.0, 101.0), (776.0, 102.0), (776.0, 135.0), (86.0, 134.0)]
    assert len(ann.geometry) == 4
    assert all(isinstance(v, float) for point in ann.geometry for v in point)
    # the TL -> TR -> BR -> BL order is preserved: the points run clockwise
    assert shoelace(ann.geometry) > 0


def test_skewed_line_stays_quadrilateral():
    """A slanted line is not straightened into an axis-aligned box."""
    (ann,) = mod.to_annotations([SKEWED], ("skewed",), (0.97,))
    assert ann.geometry == [(172.0, 99.0), (868.0, 135.0), (866.0, 181.0), (170.0, 145.0)]
    assert ann.geometry[0][1] != ann.geometry[1][1]
    assert shoelace(ann.geometry) > 0


def test_numpy_boxes_and_float32_scores():
    np = pytest.importorskip("numpy")
    boxes = np.array([BOX_A], dtype=np.float32)
    scores = np.array([0.98514], dtype=np.float32)
    (ann,) = mod.to_annotations(boxes, np.array(["line"]), scores)
    assert ann.geometry == [(86.0, 101.0), (776.0, 102.0), (776.0, 135.0), (86.0, 134.0)]
    assert ann.text == "line"
    assert ann.confidence == pytest.approx(0.98514, abs=1e-6)


def test_unreadable_line_is_kept_not_dropped():
    """Detection found a line, recognition returned nothing — the polygon stays."""
    anns = mod.to_annotations([BOX_A, BOX_B], ("", "ok"), (0.11, 0.99))
    assert len(anns) == 2
    assert anns[0].text == ""
    assert anns[0].confidence == 0.11


def test_degenerate_box_skipped():
    two_points = [[1.0, 2.0], [3.0, 4.0]]
    anns = mod.to_annotations([two_points, BOX_A], ("bad", "good"), (0.9, 0.9))
    assert [a.text for a in anns] == ["good"]


# --- the min_confidence filter ---


def test_min_confidence_filters_below_threshold():
    anns = mod.to_annotations(
        [BOX_A, BOX_B, SKEWED],
        ("keep", "drop", "keep too"),
        (0.99, 0.52, 0.9501),
        min_confidence=0.95,
    )
    assert [a.text for a in anns] == ["keep", "keep too"]


def test_min_confidence_zero_keeps_everything():
    anns = mod.to_annotations([BOX_A, BOX_B], ("a", "b"), (0.0, 0.5), min_confidence=0.0)
    assert len(anns) == 2


def test_min_confidence_is_inclusive_on_threshold():
    anns = mod.to_annotations([BOX_A], ("edge",), (0.9,), min_confidence=0.9)
    assert len(anns) == 1


# --- empty result ---


def test_empty_page_returns_empty_list():
    """Empty page: rapidocr returns boxes/txts/scores = None — not an error."""
    assert mod.to_annotations(None, None, None) == []
    assert mod.to_annotations([], (), ()) == []


def test_partial_none_is_not_an_error():
    assert mod.to_annotations([BOX_A], None, (0.9,)) == []


# --- language resolution ---


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        (None, "cyrillic"),
        ("", "cyrillic"),
        ("ru", "cyrillic"),
        ("RU", "cyrillic"),
        (" ru ", "cyrillic"),
        ("uk", "cyrillic"),
        ("en", "en"),
        ("de", "latin"),
        ("latin", "latin"),
        ("eslav", "eslav"),
        ("zh_TW", "chinese_cht"),
    ],
)
def test_resolve_rec_lang(lang, expected):
    assert mod.resolve_rec_lang(lang) == expected


def test_resolve_rec_lang_rejects_unknown():
    with pytest.raises(ValueError, match="unsupported lang"):
        mod.resolve_rec_lang("klingon")


# --- lazy engine initialization, one instance per rec model ---


def test_engine_created_once_per_rec_model(monkeypatch):
    created = []

    def fake_create(rec_lang):
        created.append(rec_lang)
        return f"engine:{rec_lang}"

    monkeypatch.setattr(RapidOCRLabeler, "_create_engine", staticmethod(fake_create))
    instance = RapidOCRLabeler()

    assert instance._engines == {}  # nothing is loaded before the first predict
    assert instance._engine({"lang": "ru"}) == "engine:cyrillic"
    assert instance._engine({"lang": "uk"}) == "engine:cyrillic"  # same model
    assert instance._engine({}) == "engine:cyrillic"  # default lang
    assert instance._engine({"lang": "en"}) == "engine:en"
    assert created == ["cyrillic", "en"]


# --- robustness: corrupt bytes and the size limit ---


def test_probe_image_returns_size():
    assert mod.probe_image(png(320, 200), 10_000_000) == (320, 200)


def test_probe_image_rejects_garbage():
    with pytest.raises(ValueError, match="cannot decode image"):
        mod.probe_image(b"definitely not an image", 10_000_000)


def test_probe_image_rejects_empty_payload():
    with pytest.raises(ValueError, match="empty image payload"):
        mod.probe_image(b"", 10_000_000)


def test_probe_image_rejects_oversized():
    with pytest.raises(ValueError, match="exceeds the limit"):
        mod.probe_image(png(300, 300), 10_000)


def test_max_pixels_config_wins_over_env(monkeypatch):
    monkeypatch.setenv(mod.ENV_MAX_PIXELS, "123")
    assert mod.max_pixels({"max_pixels": 456}) == 456
    assert mod.max_pixels({}) == 123
    monkeypatch.delenv(mod.ENV_MAX_PIXELS)
    assert mod.max_pixels({}) == mod.DEFAULT_MAX_PIXELS


def test_max_pixels_rejects_nonpositive():
    with pytest.raises(ValueError, match="must be positive"):
        mod.max_pixels({"max_pixels": -1})


def test_predict_rejects_garbage_before_loading_model(monkeypatch):
    def boom(rec_lang):
        raise AssertionError("engine must not be loaded for undecodable input")

    monkeypatch.setattr(RapidOCRLabeler, "_create_engine", staticmethod(boom))
    with pytest.raises(ValueError, match="cannot decode image"):
        RapidOCRLabeler().predict(b"\x00\x01\x02", {})


# --- weight cache ---


def test_model_dir_from_env(monkeypatch, tmp_path):
    target = tmp_path / "weights"
    monkeypatch.setenv(mod.ENV_MODEL_DIR, str(target))
    assert mod.model_dir() == str(target)
    assert target.is_dir()


def test_model_dir_falls_back_when_unwritable(monkeypatch, tmp_path):
    fallback = tmp_path / "cache"
    monkeypatch.setattr(mod, "FALLBACK_MODEL_DIR", fallback)
    monkeypatch.setenv(mod.ENV_MODEL_DIR, str(tmp_path / "file.txt" / "weights"))
    (tmp_path / "file.txt").write_text("not a directory")
    assert mod.model_dir() == str(fallback)
    assert fallback.is_dir()


# --- platform contract ---


def test_satisfies_sdk_protocol():
    instance = RapidOCRLabeler()
    assert isinstance(instance, Labeler)
    assert instance.name == "rapidocr"
    assert {c.value for c in instance.capabilities} == {"detection", "recognition"}
