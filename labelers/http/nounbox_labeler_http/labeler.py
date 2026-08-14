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

WHAT GOES INTO X-Labeler-Config. For the generic `http` engine: exactly
config["backend_config"], nothing else — the endpoint is a third party's and
the contract above is what it was written against. The platform's own engines
(modal_gpu, modal_gpu_detect) override _backend_config to also forward the
handful of keys the matching recipe understands, because the platform fills
those in itself: the worker writes the project classes into config["classes"]
and the UI writes config["score_threshold"], and neither of them knows about
the backend_config envelope. Without that forwarding the recipes received `{}`
on every image. `endpoint`, `api_key` and `timeout` are ours and never travel.

4xx VERSUS 5xx. A 4xx is raised as ValueError, deliberately: run_autolabel
treats ValueError as "this will fail the same way on every image" and stops the
job with the message, while any other exception counts one failed image and
lets the run finish DONE. A wrong bearer token or an empty class list used to
produce a cheerful "Done, 0 annotations". A 5xx keeps raise_for_status —
that one really can be a single flaky image.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

import httpx

from nounbox_sdk import Annotation, BBox, Capability

DEFAULT_ENDPOINT = os.environ.get("LABELER_HTTP_ENDPOINT", "")

# One pooled client for the whole process instead of httpx.post, which builds a
# throwaway client per call. That cost is invisible on a local endpoint and
# dominant on a remote one: every image paid a fresh TCP connection and a full
# TLS handshake to the GPU host. Measured against a Modal T4, a labeling run
# spent roughly a third of its wall clock on handshakes it did not need.
#
# httpx.Client is documented as thread-safe, which is what this needs — the
# platform calls predict() from several worker threads at once. The pool is
# sized above the platform's own concurrency so the threads never queue on a
# free connection, and keepalive expiry lets idle connections go on their own.
_LIMITS = httpx.Limits(
    max_connections=32, max_keepalive_connections=32, keepalive_expiry=90.0
)
_CLIENT: httpx.Client | None = None
_CLIENT_LOCK = threading.Lock()


def _client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:  # another thread may have won the race
                _CLIENT = httpx.Client(limits=_LIMITS)
    return _CLIENT


def _post(url: str, *, content: bytes, headers: dict, timeout: float):
    """Single seam for the request, so the pooling stays an implementation detail."""
    return _client().post(url, content=content, headers=headers, timeout=timeout)

# Headers have a size limit (nginx and uvicorn both answer 431 well before
# 64 KB). A project with hundreds of classes would hit it as an unreadable
# protocol error in the middle of a batch, so we say what is wrong instead.
MAX_CONFIG_BYTES = 8000


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

    def _backend_config(self, config: dict) -> dict:
        """What lands in X-Labeler-Config. The documented contract: the
        backend_config envelope and nothing else. Subclasses that own their
        backend widen this."""
        return dict(config.get("backend_config") or {})

    def predict(self, image: bytes, config: dict) -> list[Annotation]:
        endpoint = config.get("endpoint") or DEFAULT_ENDPOINT
        if not endpoint:
            raise ValueError(
                "http labeler: endpoint not set (config.endpoint or LABELER_HTTP_ENDPOINT)"
            )

        encoded = json.dumps(self._backend_config(config))
        size = len(encoded.encode())
        if size > MAX_CONFIG_BYTES:
            raise ValueError(
                f"{self.name}: the engine config is too large for the "
                f"X-Labeler-Config header ({size} bytes, limit {MAX_CONFIG_BYTES}); "
                "reduce the class list"
            )

        headers = {
            "Content-Type": "application/octet-stream",
            "X-Labeler-Config": encoded,
        }
        if config.get("api_key"):
            headers["Authorization"] = f"Bearer {config['api_key']}"

        response = _post(
            endpoint,
            content=image,
            headers=headers,
            timeout=float(config.get("timeout", 300)),
        )
        if 400 <= response.status_code < 500 and response.status_code != 422:
            # the same request will fail identically on every other image, so
            # this aborts the job. 422 is the deliberate exception: it means
            # THIS image was unreadable, which says nothing about the next one,
            # and it falls through to raise_for_status below — a plain HTTP
            # error the worker counts as one failed frame.
            raise ValueError(
                f"{self.name}: endpoint answered {response.status_code} — "
                f"{response.text[:500]}"
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
    """GPU OCR made turnkey: same HTTP contract, but the platform supplies the
    endpoint.

    The user configures nothing by hand: the address of the recipe deployed
    into their own Modal account comes from the installation settings
    (see services/settings_store.resolve_labeler_config). The recipe behind it
    is deploy/modal/paddleocr_modal.py — OCR, hence RECOGNITION only.
    """

    name = "modal_gpu"
    capabilities = {Capability.RECOGNITION}

    # keys of the PaddleOCR recipe that the platform may set on the engine
    # config directly; `lang` in particular has been ignored until now
    FORWARDED = (
        "lang",
        "ocr_version",
        "det_model",
        "rec_model",
        "text_det_limit_type",
        "text_det_limit_side_len",
        "text_det_thresh",
        "text_det_box_thresh",
        "text_det_unclip_ratio",
        "text_rec_score_thresh",
    )

    def _backend_config(self, config: dict) -> dict:
        payload = {k: config[k] for k in self.FORWARDED if config.get(k) is not None}
        # an explicit backend_config always wins over the forwarded keys
        payload.update(super()._backend_config(config))
        return payload


class ModalGpuDetectLabeler(HttpLabeler):
    """GPU box detection made turnkey — the GPU twin of the owlv2 CPU engine.

    Behind it is deploy/modal/owlv2_modal.py: same model, same thresholds, same
    post-processing, so a project half-labeled on CPU and half on GPU is still
    one dataset.
    """

    name = "modal_gpu_detect"
    version = "0.1.0"
    capabilities = {Capability.DETECTION}

    # what owlv2_modal.py reads out of X-Labeler-Config. `classes` is written
    # by the worker from the project classes, `score_threshold` by the UI.
    FORWARDED = ("classes", "score_threshold", "nms_iou", "model")

    def _backend_config(self, config: dict) -> dict:
        payload = {k: config[k] for k in self.FORWARDED if config.get(k) is not None}
        payload.update(super()._backend_config(config))
        return payload

    def predict(self, image: bytes, config: dict) -> list[Annotation]:
        classes = [
            str(c).strip() for c in (config.get("classes") or []) if str(c).strip()
        ]
        if not classes:
            # an empty list is an error with a hint, not "find whatever you
            # can": staying quiet ends the whole run with zero annotations
            raise ValueError(
                f"{self.name}: no classes given — add the project classes "
                "before labeling"
            )

        annotations = super().predict(image, config)

        # A label that is not a project class means the endpoint is not the
        # Nounbox detection recipe — most likely an app deployed from the OCR
        # recipe under this URL, answering with text_line boxes. Dropping them
        # silently is exactly the failure being fixed: export routes unknown
        # labels into skipped_labels and the user gets an empty dataset.
        known = set(classes)
        for annotation in annotations:
            if annotation.label not in known:
                raise ValueError(
                    f"{self.name}: the endpoint returned label "
                    f"{annotation.label!r}, which is not one of the project "
                    "classes — the app deployed in your Modal account is "
                    "probably not the Nounbox detection recipe; press Redeploy "
                    "on the settings page"
                )
        return annotations
