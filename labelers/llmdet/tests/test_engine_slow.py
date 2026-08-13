"""Runs the real model: pytest -m slow.

The weights are downloaded into OVD_MODEL_DIR on the first run (llmdet_tiny
~0.7 GB); point that variable at a cached directory, or every run in a clean
environment downloads them again. The tests use tiny — the forward pass and the
output parsing are the same as in base, at half the weights and memory.

What is checked is not "was the cat found" (a synthetic image is not a photo,
quality is measured by a separate eval set) but the integration invariants: the
prompt splits into exactly as many class spans as there are classes, a box label
is the PROJECT class name in its original case, the coordinates are pixels of
this image, and the threshold really does prune.
"""

import io

import pytest
from PIL import Image, ImageDraw

from nounbox_labeler_llmdet import LLMDetLabeler
from nounbox_labeler_llmdet import labeler as mod
from nounbox_sdk import BBox

pytestmark = pytest.mark.slow

CLASSES = ["Cat", "Traffic Light", "Wooden Spoon"]

# exactly MAX_CLASSES real one-word names: eval set vocabulary plus household items
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
    """Colorful synthetic scene: something for the model to latch onto, not a photo."""
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


# --- the reason the limit exists: the model's real numbers ---


def test_model_text_limit_matches_our_constant(engine):
    """A max_text_len change in transformers/weights surfaces here, not in prod."""
    assert engine.max_text_len == mod.DEFAULT_MAX_TEXT_TOKENS


def test_max_classes_of_typical_names_fit_the_prompt(engine):
    """MAX_CLASSES one-word classes fit in 256 tokens — the limit is not a guess."""
    assert len(TYPICAL_CLASSES) == mod.MAX_CLASSES
    tokens = engine.processor.tokenizer(mod.build_prompt(TYPICAL_CLASSES))["input_ids"]
    assert len(tokens) <= engine.max_text_len  # 212 of 256


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
    """The same 91 classes, but two-word, no longer fit — hence the second gate."""
    classes = [f"wooden {name}" for name in TYPICAL_CLASSES]
    tokens = engine.processor.tokenizer(mod.build_prompt(classes))["input_ids"]
    assert len(tokens) > engine.max_text_len  # exactly the case we guard against
    with pytest.raises(ValueError, match="tokens"):
        engine.detect(Image.new("RGB", (64, 64)), classes, 0.35)


# --- the forward pass ---


def test_predict_returns_valid_detection_annotations(instance):
    image = scene()
    annotations = instance.predict(
        image, {"classes": CLASSES, "model": "tiny", "score_threshold": 0.05}
    )

    for annotation in annotations:
        assert isinstance(annotation.geometry, BBox)
        assert annotation.label in CLASSES  # the project classes' original case
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
    assert low  # at a 0.05 threshold the model always proposes something
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
    assert instance._engines == loaded  # same objects: there was no second load
