"""VLM labeler: страница целиком -> боксы + текст через OpenAI-compatible API.

Работает с Ollama (qwen2.5vl, minicpm-v...), vLLM, OpenAI и любыми
совместимыми endpoint'ами — меняется только base_url/model в config.

Разрешение настроек: config -> переменная окружения -> дефолт:
    base_url:  env VLM_BASE_URL, default http://host.docker.internal:11434/v1
    api_key:   env VLM_API_KEY, default "ollama"
    model:     env VLM_MODEL, default "qwen2.5vl:7b"
    reported_confidence: float, default 0.5
    max_tokens: int, default 4096
    temperature: float, default 0.0

api_key задаётся через VLM_API_KEY на сервере, а не в UI-конфиге:
config попадает в Job.payload и оседает в БД открытым текстом.

VLM не возвращает confidence — проставляется синтетический reported_confidence.
Дефолт 0.5 намеренно ниже типового порога bulk-accept: разметка VLM должна
пройти ревью человеком.
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading

from PIL import Image

from nounbox_sdk import Annotation, Capability

DEFAULT_BASE_URL = "http://host.docker.internal:11434/v1"
DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_API_KEY = "ollama"

PROMPT = """Extract all text from this image. Return ONLY a JSON array, no markdown fences:
[{{"bbox": [x1, y1, x2, y2], "text": "..."}}]
- bbox in absolute pixels; the image is {width}x{height}
- one entry per text line, in reading order
- preserve the original language, case and punctuation
- if there is no text, return []"""


class VlmLabeler:
    name = "vlm"
    version = "0.1.0"
    capabilities = {Capability.DETECTION, Capability.RECOGNITION}

    def __init__(self) -> None:
        self._clients: dict[tuple[str, str], object] = {}
        self._lock = threading.Lock()

    def _client(self, base_url: str, api_key: str):
        key = (base_url, api_key)
        with self._lock:
            if key not in self._clients:
                from openai import OpenAI

                self._clients[key] = OpenAI(
                    base_url=base_url, api_key=api_key, timeout=300.0
                )
            return self._clients[key]

    def predict(self, image: bytes, config: dict) -> list[Annotation]:
        base_url = (
            config.get("base_url") or os.environ.get("VLM_BASE_URL") or DEFAULT_BASE_URL
        )
        api_key = (
            config.get("api_key") or os.environ.get("VLM_API_KEY") or DEFAULT_API_KEY
        )
        model = config.get("model") or os.environ.get("VLM_MODEL") or DEFAULT_MODEL
        confidence = float(config.get("reported_confidence", 0.5))

        with Image.open(io.BytesIO(image)) as pil:
            width, height = pil.size
        mime = self._sniff_mime(image)
        b64 = base64.b64encode(image).decode()

        client = self._client(base_url, api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 4096)),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT.format(width=width, height=height),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        if not response.choices:
            raise ValueError(
                f"VLM returned no choices (model={model!r}, base_url={base_url!r})"
            )
        content = response.choices[0].message.content or ""
        return self._parse(content, width, height, confidence)

    @staticmethod
    def _sniff_mime(image: bytes) -> str:
        """MIME по магическим байтам; ingest нормализует в PNG — это и дефолт."""
        if image[:4] == b"\x89PNG":
            return "image/png"
        if image[:2] == b"\xff\xd8":
            return "image/jpeg"
        if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
            return "image/webp"
        return "image/png"

    @staticmethod
    def _strip_fences(content: str) -> str:
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content.strip("`")
            content = content.rstrip()
            if content.endswith("```"):
                content = content[:-3]
        return content

    @staticmethod
    def _extract_array(content: str) -> list | None:
        """Первый полный top-level JSON-массив.

        Сбалансированный скан с учётом строк и экранирования: жадный
        regex ловил бы всё от первой '[' до последней ']' и ломался,
        когда после массива идёт проза со скобками или второй массив.
        Кандидаты, не являющиеся валидным JSON (скобки в прозе),
        пропускаются. Массив без dict-элементов (например, голые
        координаты в прозе) — только fallback: настоящий массив
        аннотаций может идти дальше по тексту.
        """
        fallback = None
        start = content.find("[")
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for i in range(start, len(content)):
                ch = content[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                elif ch == '"':
                    in_string = True
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            candidate = json.loads(content[start : i + 1])
                        except json.JSONDecodeError:
                            break
                        if not candidate or any(
                            isinstance(el, dict) for el in candidate
                        ):
                            return candidate
                        if fallback is None:
                            fallback = candidate
                        break
            start = content.find("[", start + 1)
        return fallback

    @staticmethod
    def _parse(
        content: str, width: int, height: int, confidence: float
    ) -> list[Annotation]:
        """Извлечь JSON-массив из ответа, отвалидировать и обрезать боксы."""
        data = VlmLabeler._extract_array(VlmLabeler._strip_fences(content))
        if data is None:
            raise ValueError(f"VLM returned no JSON array: {content[:200]!r}")

        annotations = []
        for item in data:
            try:
                x1, y1, x2, y2 = (float(v) for v in item["bbox"])
            except (KeyError, TypeError, ValueError):
                continue
            x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
            y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
            text = str(item.get("text", "")).strip()
            if x2 - x1 < 2 or y2 - y1 < 2 or not text:
                continue
            annotations.append(
                Annotation(
                    geometry=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                    label="text_line",
                    text=text,
                    confidence=confidence,
                )
            )
        return annotations
