"""The labeler plugin protocol and its loader."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nounbox_sdk.types import Annotation, Capability

ENTRYPOINT_GROUP = "nounbox.labelers"


@runtime_checkable
class Labeler(Protocol):
    """The auto-labeling engine contract.

    An implementation is an ordinary class, no base class required:

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
        """Label a single image.

        :param image: image bytes (PNG/JPEG — normalized at ingest, not the raw upload)
        :param config: user settings for the run (language, thresholds, etc.)
        :return: annotations with confidence; source is filled in by the platform
        """
        ...


def load_labelers() -> dict[str, Labeler]:
    """Find every installed labeler through entry points."""
    from importlib.metadata import entry_points

    found: dict[str, Labeler] = {}
    for ep in entry_points(group=ENTRYPOINT_GROUP):
        try:
            instance = ep.load()()
            found[instance.name] = instance
        except Exception as exc:  # a plugin must not bring the platform down
            import logging

            logging.getLogger(__name__).warning(
                "Failed to load labeler %r: %s", ep.name, exc
            )
    return found
