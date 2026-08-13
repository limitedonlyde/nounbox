"""Протокол labeler-плагина и загрузчик."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nounbox_sdk.types import Annotation, Capability

ENTRYPOINT_GROUP = "nounbox.labelers"


@runtime_checkable
class Labeler(Protocol):
    """Контракт движка авторазметки.

    Реализация — обычный класс без обязательного наследования:

        class MyLabeler:
            name = "my-ocr"
            version = "0.1.0"
            capabilities = {Capability.DETECTION, Capability.RECOGNITION}

            def predict(self, image: bytes, config: dict) -> list[Annotation]:
                ...
    """

    name: str
    version: str
    capabilities: set[Capability]

    def predict(self, image: bytes, config: dict) -> list[Annotation]:
        """Разметить одно изображение.

        :param image: байты изображения (JPEG/PNG — как загружено после нормализации)
        :param config: пользовательские настройки запуска (язык, пороги и т.п.)
        :return: список аннотаций с confidence; source проставит платформа
        """
        ...


def load_labelers() -> dict[str, Labeler]:
    """Найти все установленные labeler'ы через entry points."""
    from importlib.metadata import entry_points

    found: dict[str, Labeler] = {}
    for ep in entry_points(group=ENTRYPOINT_GROUP):
        try:
            instance = ep.load()()
            found[instance.name] = instance
        except Exception as exc:  # плагин не должен ронять платформу
            import logging

            logging.getLogger(__name__).warning(
                "Failed to load labeler %r: %s", ep.name, exc
            )
    return found
