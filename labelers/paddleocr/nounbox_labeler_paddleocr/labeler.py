"""PaddleOCR labeler: text line detection + recognition.

Both generations of the PaddleOCR API (2.x and 3.x) are supported — the choice
follows the installed major version. Models are downloaded on the first predict
and cached in ~/.paddleocr (in docker — the paddleocr-data volume).

Config (passed in from the API / the autolabel job):
    lang: str = "en"   — model language ("en", "ch", "ru", ...)
"""

from __future__ import annotations

import io
import logging
import threading
from importlib.metadata import version as pkg_version

import numpy as np
from PIL import Image

from nounbox_sdk import Annotation, Capability

logger = logging.getLogger(__name__)


class PaddleOCRLabeler:
    name = "paddleocr"
    version = "0.1.0"
    capabilities = {Capability.DETECTION, Capability.RECOGNITION}

    def __init__(self) -> None:
        self._engines: dict[str, object] = {}  # lang -> PaddleOCR instance
        self._lock = threading.Lock()

    # --- engine initialization (lazily, once per language) ---

    def _engine(self, config: dict):
        lang = config.get("lang", "en")
        with self._lock:
            if lang not in self._engines:
                self._engines[lang] = self._create_engine(lang)
            return self._engines[lang]

    @staticmethod
    def _create_engine(lang: str):
        from paddleocr import PaddleOCR

        major = int(pkg_version("paddleocr").split(".", 1)[0])
        logger.info("Loading PaddleOCR %s (lang=%s)...", pkg_version("paddleocr"), lang)
        if major >= 3:
            # lightweight pipeline: no orientation/unwarping models
            return PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return PaddleOCR(lang=lang, use_angle_cls=False, show_log=False)

    # --- SDK contract ---

    def predict(self, image: bytes, config: dict) -> list[Annotation]:
        ocr = self._engine(config)
        arr = np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))

        major = int(pkg_version("paddleocr").split(".", 1)[0])
        rows = self._run_3x(ocr, arr) if major >= 3 else self._run_2x(ocr, arr)

        return [
            Annotation(
                geometry=points,
                label="text_line",
                text=text,
                confidence=confidence,
            )
            for points, text, confidence in rows
        ]

    @staticmethod
    def _run_2x(ocr, arr: np.ndarray) -> list[tuple[list, str, float]]:
        """PaddleOCR 2.x: ocr.ocr() -> [[box, (text, conf)], ...]"""
        result = ocr.ocr(arr, cls=False)
        lines = (result or [None])[0] or []
        return [
            ([(float(x), float(y)) for x, y in box], str(text), float(conf))
            for box, (text, conf) in lines
        ]

    @staticmethod
    def _run_3x(ocr, arr: np.ndarray) -> list[tuple[list, str, float]]:
        """PaddleOCR 3.x: predict() -> [OCRResult with rec_texts/rec_scores/dt_polys]"""
        results = ocr.predict(input=arr)
        rows = []
        for res in results or []:
            data = res.json if hasattr(res, "json") else res
            # res.json is {"res": {...}} in newer versions; otherwise res is dict-like
            inner = data.get("res", data) if isinstance(data, dict) else data
            texts = inner["rec_texts"]
            scores = inner["rec_scores"]
            polys = inner.get("dt_polys", inner.get("rec_polys"))
            for poly, text, score in zip(polys, texts, scores):
                points = [(float(x), float(y)) for x, y in np.asarray(poly)]
                rows.append((points, str(text), float(score)))
        return rows
