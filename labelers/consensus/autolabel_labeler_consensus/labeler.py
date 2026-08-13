"""Consensus meta-labeler: реальная уверенность из согласия движков.

Прогоняет изображение через несколько установленных движков по порядку;
первый в списке — primary (его геометрия точнее, обычно OCR), остальные
подтверждают или дополняют. Сведение — merge.merge(): IoU-матчинг
геометрий + сходство текста -> честный confidence вместо синтетического.

Config:
    engines: [{"name": "paddleocr", "config": {...}}, {"name": "vlm", "config": {...}}]
        default: [{"name": "paddleocr"}, {"name": "vlm"}]
    iou_threshold: float = 0.4 — порог геометрического совпадения
"""

from __future__ import annotations

from autolabelui_sdk import Annotation, Capability, load_labelers

from autolabel_labeler_consensus.merge import DEFAULT_IOU_THRESHOLD, merge

DEFAULT_ENGINES = [{"name": "paddleocr"}, {"name": "vlm"}]


class ConsensusLabeler:
    name = "consensus"
    version = "0.1.0"
    capabilities = {Capability.DETECTION, Capability.RECOGNITION}

    def predict(self, image: bytes, config: dict) -> list[Annotation]:
        engine_specs = config.get("engines", DEFAULT_ENGINES)
        if not engine_specs:
            raise ValueError('consensus: "engines" must be a non-empty list')
        iou_threshold = float(config.get("iou_threshold", DEFAULT_IOU_THRESHOLD))

        available = load_labelers()
        annotations_by_engine: dict[str, list[Annotation]] = {}
        for spec in engine_specs:
            if isinstance(spec, str):
                spec = {"name": spec}
            engine_name = spec.get("name")
            if not engine_name:
                raise ValueError('consensus: each engines entry needs a "name"')
            if engine_name == self.name:
                raise ValueError("consensus: cannot include itself in engines")
            labeler = available.get(engine_name)
            if labeler is None:
                installed = ", ".join(sorted(available)) or "<none>"
                raise ValueError(
                    f"consensus: engine {engine_name!r} is not installed; "
                    f"available: {installed}"
                )
            key = engine_name
            suffix = 2
            while key in annotations_by_engine:  # один движок дважды (разные config)
                key = f"{engine_name}#{suffix}"
                suffix += 1
            # общие настройки задачи (classes от платформы, пороги) уходят
            # каждому движку; явный config движка их перекрывает
            shared = {
                k: v for k, v in config.items() if k not in ("engines", "iou_threshold")
            }
            sub_config = {**shared, **(spec.get("config") or {})}
            annotations_by_engine[key] = labeler.predict(image, sub_config)
        return merge(annotations_by_engine, iou_threshold=iou_threshold)
