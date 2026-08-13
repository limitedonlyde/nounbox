"""Generic HTTP labeler: any engine behind a tiny HTTP convention.

Lets you move GPU inference anywhere (Modal, RunPod, your own server) without
writing a plugin for the platform — it is enough to stand up an endpoint:

    POST {endpoint}
    Body: image bytes (Content-Type: application/octet-stream)
    Headers: X-Labeler-Config: <json> — config["backend_config"], passed through
             Authorization: Bearer <api_key> — if config["api_key"] is set

    200 response: {"annotations": [
        {"geometry": {"type": "bbox", "x":.., "y":.., "width":.., "height":..}
                    | {"type": "polygon", "points": [[x, y], ...]},
         "label": "text_line", "text": "...", "confidence": 0.97}
    ]}

Config: endpoint (or env LABELER_HTTP_ENDPOINT), api_key, backend_config, timeout.
Ready-made backends live in deploy/modal/ of the monorepo.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from nounbox_sdk import Annotation, BBox, Capability

DEFAULT_ENDPOINT = os.environ.get("LABELER_HTTP_ENDPOINT", "")


class HttpLabeler:
    name = "http"
    version = "0.1.0"
    # what the backend can actually do is unknown; we declare the maximum,
    # the real capabilities are whatever the endpoint answers with
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
                "http labeler: endpoint not set (config.endpoint or LABELER_HTTP_ENDPOINT)"
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
    """The "hard" mode made turnkey: same HTTP contract, but the platform
    supplies the endpoint.

    The user configures nothing by hand: the address of the recipe deployed
    into their own Modal account comes from the installation settings
    (see services/settings_store.resolve_labeler_config).
    """

    name = "modal_gpu"
