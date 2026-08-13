"""Тесты robust-парсера VLM-ответов — без сети, только чистые функции."""

import json

import pytest

from nounbox_labeler_vlm import VlmLabeler

WIDTH, HEIGHT = 1000, 800
CONF = 0.5


def parse(content: str):
    return VlmLabeler._parse(content, WIDTH, HEIGHT, CONF)


def box(ann):
    xs = [p[0] for p in ann.geometry]
    ys = [p[1] for p in ann.geometry]
    return (min(xs), min(ys), max(xs), max(ys))


def test_clean_json():
    anns = parse('[{"bbox": [10, 20, 110, 40], "text": "hello"}]')
    assert len(anns) == 1
    assert anns[0].text == "hello"
    assert anns[0].label == "text_line"
    assert anns[0].confidence == CONF
    assert box(anns[0]) == (10.0, 20.0, 110.0, 40.0)


def test_markdown_fences():
    content = '```json\n[{"bbox": [10, 20, 110, 40], "text": "fenced"}]\n```'
    anns = parse(content)
    assert len(anns) == 1
    assert anns[0].text == "fenced"


def test_prose_around_array_with_brackets():
    # Ломало жадный regex: он захватывал от первой '[' до последней ']'
    content = (
        "Here is the result [as requested]:\n"
        '[{"bbox": [10, 20, 110, 40], "text": "line"}]\n'
        "Let me know [if] anything is missing."
    )
    anns = parse(content)
    assert len(anns) == 1
    assert anns[0].text == "line"


def test_two_arrays_takes_first():
    content = (
        '[{"bbox": [10, 20, 110, 40], "text": "first"}]\n'
        'And corrected version:\n'
        '[{"bbox": [10, 20, 110, 40], "text": "second"}]'
    )
    anns = parse(content)
    assert len(anns) == 1
    assert anns[0].text == "first"


def test_brackets_inside_string_values():
    payload = [{"bbox": [10, 20, 110, 40], "text": 'arr[0] = "x]" \\ [done'}]
    anns = parse(json.dumps(payload) + " trailing ] prose [")
    assert len(anns) == 1
    assert anns[0].text == 'arr[0] = "x]" \\ [done'


def test_clamp_out_of_bounds():
    anns = parse('[{"bbox": [-50, -10, 2000, 900], "text": "big"}]')
    assert len(anns) == 1
    assert box(anns[0]) == (0.0, 0.0, float(WIDTH), float(HEIGHT))


def test_inverted_coordinates():
    anns = parse('[{"bbox": [110, 40, 10, 20], "text": "flipped"}]')
    assert len(anns) == 1
    assert box(anns[0]) == (10.0, 20.0, 110.0, 40.0)


def test_garbage_items_skipped():
    content = json.dumps(
        [
            {"bbox": "nope", "text": "bad bbox"},
            {"text": "no bbox"},
            {"bbox": [0, 0, 100, 30, 5], "text": "five coords"},
            {"bbox": [0, 0, 1, 1], "text": "tiny"},
            {"bbox": [0, 0, 100, 30], "text": "   "},
            "not a dict",
            {"bbox": [10, 20, 110, 40], "text": "ok"},
        ]
    )
    anns = parse(content)
    assert len(anns) == 1
    assert anns[0].text == "ok"


def test_empty_array():
    assert parse("[]") == []
    assert parse("No text found, so: []") == []


def test_no_array_raises_value_error():
    with pytest.raises(ValueError, match="no JSON array"):
        parse("Sorry, I cannot see any text in this image.")


def test_sniff_mime():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 8
    webp = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 8
    assert VlmLabeler._sniff_mime(png) == "image/png"
    assert VlmLabeler._sniff_mime(jpeg) == "image/jpeg"
    assert VlmLabeler._sniff_mime(webp) == "image/webp"
    assert VlmLabeler._sniff_mime(b"unknown bytes") == "image/png"


def test_empty_choices_raises_value_error(monkeypatch):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="PNG")

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            class Response:
                choices = []

            return Response()

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    labeler = VlmLabeler()
    monkeypatch.setattr(labeler, "_client", lambda base_url, api_key: FakeClient())
    with pytest.raises(ValueError, match="no choices"):
        labeler.predict(buf.getvalue(), {})


def test_single_line_fenced_array():
    assert parse("```json []```") == []


def test_bare_coordinate_list_before_real_array():
    content = (
        "Found a region at [10, 20, 30, 40].\n"
        '[{"bbox": [5, 5, 50, 25], "text": "real"}]'
    )
    assert [a.text for a in parse(content)] == ["real"]


def test_only_bare_arrays_yield_no_annotations():
    assert parse("data: [1, 2, 3] and [4, 5]") == []
