"""Прогон реальной модели: pytest -m slow.

Веса качаются в OVD_MODEL_DIR при первом запуске (llmdet_tiny ~0.7 ГБ); задайте
эту переменную на кешируемый каталог, иначе каждый прогон в чистом окружении
будет качать их заново. Тесты идут на tiny — форвард и разбор выхода у base
те же, а весов и памяти вдвое меньше.

Проверяется не «нашлась ли кошка» (синтетическая картинка — не фотография,
качество меряется отдельным eval-набором), а инварианты интеграции: промпт
режется на классы ровно по числу классов, ярлык рамки — имя класса ПРОЕКТА
в исходном регистре, координаты в пикселях этой картинки, порог реально режет.
"""

import io

import pytest
from PIL import Image, ImageDraw

from autolabel_labeler_llmdet import LLMDetLabeler
from autolabel_labeler_llmdet import labeler as mod
from autolabelui_sdk import BBox

pytestmark = pytest.mark.slow

CLASSES = ["Cat", "Traffic Light", "Wooden Spoon"]

# ровно MAX_CLASSES реальных однословных имён: словарь eval-набора плюс бытовые предметы
TYPICAL_CLASSES = """
airplane armchair ashtray backpack banana bed bench bicycle bird bookcase bowl
broom bus cabinet car chair chandelier clock cup dishwasher dog doormat dumpster
elephant faucet fork globe handbag headboard horse kettle keyboard knife ladle
lamp lampshade laptop mirror motorcycle oven pillow pizza plate refrigerator sink
sofa stepladder stool table tablecloth teacup television toilet truck umbrella
vase wreath carpet curtain radiator sponge towel basket blender bucket candle
cushion desk fireplace fridge guitar helmet jacket ladder mug notebook piano
printer scooter shelf shoe skateboard suitcase surfboard toaster tripod vacuum
wallet watch window zebra
""".split()


def scene(width=640, height=480):
    """Пёстрая синтетика: модели есть за что зацепиться, но это не фотография."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 40, 240, 300], fill="#3b82f6")
    draw.ellipse([300, 80, 520, 300], fill="#ef4444")
    draw.rectangle([80, 340, 560, 440], fill="#22c55e")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def instance():
    return LLMDetLabeler()


@pytest.fixture(scope="module")
def engine(instance):
    return instance._engine(mod.MODEL_ALIASES["tiny"])


# --- то, ради чего стоит лимит: реальные числа модели ---


def test_model_text_limit_matches_our_constant(engine):
    """Если transformers/веса поменяют max_text_len — узнаем здесь, а не в проде."""
    assert engine.max_text_len == mod.DEFAULT_MAX_TEXT_TOKENS


def test_max_classes_of_typical_names_fit_the_prompt(engine):
    """MAX_CLASSES однословных классов влезают в 256 токенов — лимит не с потолка."""
    assert len(TYPICAL_CLASSES) == mod.MAX_CLASSES
    tokens = engine.processor.tokenizer(mod.build_prompt(TYPICAL_CLASSES))["input_ids"]
    assert len(tokens) <= engine.max_text_len  # 212 из 256


def test_prompt_splits_into_exactly_one_span_per_class(engine):
    classes = ["carpet", "cutting board", "chandelier"]
    tokens = engine.processor.tokenizer(mod.build_prompt(classes))["input_ids"]
    spans = mod.class_spans(tokens, engine.dot_id, engine.special_ids, len(classes))
    assert len(spans) == len(classes)
    decode = engine.processor.tokenizer.convert_ids_to_tokens
    assert [decode([tokens[i] for i in span]) for span in spans] == [
        ["carpet"],
        ["cutting", "board"],
        ["chan", "##del", "##ier"],
    ]


def test_too_long_prompt_raises_before_the_forward(engine):
    """Те же 91 класса, но двусловные, уже не влезают — ради этого второй рубеж."""
    classes = [f"wooden {name}" for name in TYPICAL_CLASSES]
    tokens = engine.processor.tokenizer(mod.build_prompt(classes))["input_ids"]
    assert len(tokens) > engine.max_text_len  # именно этот случай мы и защищаем
    with pytest.raises(ValueError, match="токенов"):
        engine.detect(Image.new("RGB", (64, 64)), classes, 0.35)


# --- форвард ---


def test_predict_returns_valid_detection_annotations(instance):
    image = scene()
    annotations = instance.predict(
        image, {"classes": CLASSES, "model": "tiny", "score_threshold": 0.05}
    )

    for annotation in annotations:
        assert isinstance(annotation.geometry, BBox)
        assert annotation.label in CLASSES  # исходный регистр классов проекта
        assert annotation.text is None
        assert 0.0 < annotation.confidence <= 1.0
        assert annotation.geometry.width >= mod.MIN_BOX_SIDE
        assert annotation.geometry.height >= mod.MIN_BOX_SIDE
        assert 0.0 <= annotation.geometry.x
        assert 0.0 <= annotation.geometry.y
        assert annotation.geometry.x + annotation.geometry.width <= 640
        assert annotation.geometry.y + annotation.geometry.height <= 480
    assert [a.confidence for a in annotations] == sorted(
        (a.confidence for a in annotations), reverse=True
    )


def test_score_threshold_prunes_real_output(instance):
    image = scene()
    low = instance.predict(
        image, {"classes": CLASSES, "model": "tiny", "score_threshold": 0.05}
    )
    high = instance.predict(
        image, {"classes": CLASSES, "model": "tiny", "score_threshold": 0.9}
    )
    assert low  # на пороге 0.05 модель всегда что-то предлагает
    assert len(high) <= len(low)
    assert all(a.confidence >= 0.9 for a in high)


def test_max_detections_caps_real_output(instance):
    annotations = instance.predict(
        scene(),
        {
            "classes": CLASSES,
            "model": "tiny",
            "score_threshold": 0.01,
            "max_detections": 2,
        },
    )
    assert len(annotations) == 2


def test_engine_is_loaded_once_across_predicts(instance):
    instance.predict(scene(), {"classes": ["Cat"], "model": "tiny"})
    loaded = dict(instance._engines)
    instance.predict(scene(), {"classes": ["Cat"], "model": "tiny"})
    assert instance._engines == loaded  # те же объекты: второй загрузки не было
