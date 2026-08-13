"""Generic HTTP labeler: любой движок за крошечной HTTP-конвенцией.

Позволяет выносить GPU-инференс куда угодно (Modal, RunPod, свой сервер)
без написания плагина под платформу — достаточно поднять endpoint:

    POST {endpoint}
    Body: байты изображения (Content-Type: application/octet-stream)
    Headers: X-Labeler-Config: <json> — сквозной конфиг из config["backend_config"]
             Authorization: Bearer <api_key> — если задан config["api_key"]

    Ответ 200: {"annotations": [
        {"geometry": {"type": "bbox", "x":.., "y":.., "width":.., "height":..}
                    | {"type": "polygon", "points": [[x, y], ...]},
         "label": "text_line", "text": "...", "confidence": 0.97}
    ]}

Config: endpoint (или env LABELER_HTTP_ENDPOINT), api_key, backend_config, timeout.
Готовые backend'ы — в deploy/modal/ монорепо.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from autolabelui_sdk import Annotation, BBox, Capability

DEFAULT_ENDPOINT = os.environ.get("LABELER_HTTP_ENDPOINT", "")


class HttpLabeler:
    name = "http"
    version = "0.1.0"
    # что именно умеет backend — неизвестно; декларируем максимум,
    # реальные capabilities определяются ответом endpoint'а
    capabilities = {
        Capability.DETECTION,
        Capability.RECOGNITION,
        Capability.LAYOUT,
        Capability.KIE,
    }

    def predict(self, image: bytes, config: dict) -> list[Annotation]:
        endpoint = config.get("endpoint") or DEFAULT_ENDPOINT
        if not endpoint:
            raise ValueError(
                "http labeler: endpoint not set (config.endpoint или LABELER_HTTP_ENDPOINT)"
            )

        headers = {
            "Content-Type": "application/octet-stream",
            "X-Labeler-Config": json.dumps(config.get("backend_config", {})),
        }
        if config.get("api_key"):
            headers["Authorization"] = f"Bearer {config['api_key']}"

        response = httpx.post(
            endpoint,
            content=image,
            headers=headers,
            timeout=float(config.get("timeout", 300)),
        )
        response.raise_for_status()
        return self._parse(response.json())

    @staticmethod
    def _parse(payload: dict[str, Any]) -> list[Annotation]:
        annotations = []
        for item in payload.get("annotations", []):
            geometry = item.get("geometry", {})
            if geometry.get("type") == "bbox":
                geom: BBox | list = BBox(
                    x=float(geometry["x"]),
                    y=float(geometry["y"]),
                    width=float(geometry["width"]),
                    height=float(geometry["height"]),
                )
            elif geometry.get("type") == "polygon":
                geom = [(float(x), float(y)) for x, y in geometry["points"]]
            else:
                continue
            annotations.append(
                Annotation(
                    geometry=geom,
                    label=str(item.get("label", "text_line")),
                    text=item.get("text"),
                    confidence=float(item.get("confidence", 1.0)),
                    attrs=dict(item.get("attrs", {})),
                )
            )
        return annotations


class ModalGpuLabeler(HttpLabeler):
    """Режим «сложно»: тот же HTTP-контракт, но endpoint подставляет платформа.

    Пользователь ничего не настраивает руками: адрес развёрнутого в его
    аккаунте Modal рецепта приходит из настроек инсталляции
    (см. services/settings_store.resolve_labeler_config).
    """

    name = "modal_gpu"
