"""Прогон реальной модели: pytest -m slow.

Веса (три onnx, ~13.5 МБ для русского) качаются в RAPIDOCR_MODEL_DIR при первом
запуске; задайте эту переменную на кешируемый каталог, иначе каждый прогон
в чистом окружении будет качать их заново.
"""

import io

import pytest
from PIL import Image, ImageDraw, ImageFont

from nounbox_labeler_rapidocr import RapidOCRLabeler

pytestmark = pytest.mark.slow

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

LATIN_LINES = ["HELLO WORLD", "invoice 2026", "total 45000.00"]
CYRILLIC_LINES = ["АКТ ВЫПОЛНЕННЫХ РАБОТ", "Исполнитель: ООО Ромашка", "Итого: 45 000,00"]


def load_font(size=40):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    pytest.skip("no truetype font available to render a test page")


def render(lines, font, width=1000):
    height = 80 + 70 * len(lines)
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    for index, line in enumerate(lines):
        draw.text((60, 40 + 70 * index), line, fill="black", font=font)
    buffer = io.BytesIO()
    page.save(buffer, format="PNG")
    return buffer.getvalue()


def has_glyph(font, char):
    """Глиф реально рисуется: не пустота и не tofu-квадрат вместо символа."""

    def rendered(text):
        probe = Image.new("L", (120, 80), 255)
        ImageDraw.Draw(probe).text((5, 5), text, fill=0, font=font)
        return probe.tobytes()

    return rendered(char) not in (rendered(""), rendered("\ue000"))


def blank_page(width=800, height=600):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def instance():
    return RapidOCRLabeler()


def test_blank_page_returns_no_annotations(instance):
    assert instance.predict(blank_page(), {"lang": "ru"}) == []


def test_latin_page_lines(instance):
    font = load_font()
    annotations = instance.predict(render(LATIN_LINES, font), {"lang": "en"})

    assert len(annotations) >= len(LATIN_LINES)  # построчно, не одним блоком
    for annotation in annotations:
        assert annotation.label == "text_line"
        assert len(annotation.geometry) == 4
        assert 0.0 < annotation.confidence <= 1.0
    assert len({a.confidence for a in annotations}) > 1  # confidence не константа
    assert "HELLO" in " ".join(a.text.upper() for a in annotations)


def test_cyrillic_page_lines(instance):
    font = load_font()
    if not has_glyph(font, "П"):
        pytest.skip("available font has no cyrillic glyphs")

    annotations = instance.predict(render(CYRILLIC_LINES, font), {"lang": "ru"})

    assert len(annotations) >= len(CYRILLIC_LINES)
    text = " ".join(a.text.upper() for a in annotations)
    assert "АКТ" in text
    assert "ИСПОЛНИТЕЛЬ" in text
    assert min(a.confidence for a in annotations) > 0.5

    # полигоны построчные: у каждой строки своя вертикальная полоса
    tops = sorted(min(y for _, y in a.geometry) for a in annotations)
    assert tops[-1] - tops[0] > 70


def test_min_confidence_prunes_real_output(instance):
    font = load_font()
    page = render(LATIN_LINES, font)
    everything = instance.predict(page, {"lang": "en"})
    pruned = instance.predict(page, {"lang": "en", "min_confidence": 1.01})
    assert everything
    assert pruned == []
