"""Ingest: normalization of uploaded documents into images.

Supported inputs:
- raster images (JPEG/PNG/WebP/BMP/GIF..., including multi-page TIFF/GIF)
- HEIC/HEIF (via pillow-heif)
- PDF (each page -> an image, via pypdfium2)
- ZIP archives holding any of the above (strictly one level: a ZIP inside a
  ZIP is skipped, with a warning and a counter)

Every page/frame is normalized to PNG, plus we compute a dedup hash
(SHA-256 of the normalized bytes — exact duplicates; a perceptual hash for
near-duplicates is TODO for v0.2, dHash collides on white documents)
and a blur score (variance of Laplacian — the lower, the blurrier).
"""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pillow_heif
import pypdfium2 as pdfium
from PIL import Image as PILImage
from PIL import ImageSequence

pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)

MAX_DIMENSION = 4096  # long side; anything larger gets downscaled
PDF_SCALE = 2.0  # renders PDFs at ~144 dpi
ZIP_MAX_ENTRIES = 1000
MAX_ARCHIVE_DEPTH = 1  # ZIP inside ZIP is not unpacked (a zip quine hung the worker)
ZIP_MAX_UNCOMPRESSED = 1 * 1024**3  # 1 GiB uncompressed in total per archive


class UnsupportedFormatError(ValueError):
    pass


class ArchiveTooLargeError(ValueError):
    pass


@dataclass
class NormalizedPage:
    page_index: int
    source_name: str  # source file name (for pages from an archive — the entry)
    data: bytes  # PNG
    width: int
    height: int
    content_hash: str  # SHA-256 of the normalized bytes (dedup key)
    blur_score: float


@dataclass
class ExtractResult:
    pages: list[NormalizedPage] = field(default_factory=list)
    nested_archives_skipped: int = 0


def _content_hash(png_data: bytes) -> str:
    """SHA-256 of the normalized bytes — dedup of exact duplicates."""
    return hashlib.sha256(png_data).hexdigest()


def _blur_score(img: PILImage.Image) -> float:
    """Variance of Laplacian. < ~100 usually means a blurry frame."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def _normalize(img: PILImage.Image, page_index: int, source_name: str) -> NormalizedPage:
    if img.mode != "RGB":
        # flatten transparency onto a white background
        background = PILImage.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        background.paste(rgba, mask=rgba.getchannel("A"))
        img = background

    if max(img.size) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(img.size)
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)), PILImage.LANCZOS
        )

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    png_data = buffer.getvalue()
    return NormalizedPage(
        page_index=page_index,
        source_name=source_name,
        data=png_data,
        width=img.width,
        height=img.height,
        content_hash=_content_hash(png_data),
        blur_score=_blur_score(img),
    )


def _from_pil(data: bytes, source_name: str) -> list[NormalizedPage]:
    img = PILImage.open(io.BytesIO(data))
    frames = [f.copy() for f in ImageSequence.Iterator(img)]
    return [_normalize(frame, i, source_name) for i, frame in enumerate(frames)]


def _from_pdf(data: bytes, source_name: str) -> list[NormalizedPage]:
    pdf = pdfium.PdfDocument(data)
    try:
        pages = []
        for i in range(len(pdf)):
            bitmap = pdf[i].render(scale=PDF_SCALE)
            pages.append(_normalize(bitmap.to_pil(), i, source_name))
        return pages
    finally:
        pdf.close()


def _looks_like_zip(filename: str, data: bytes) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext == "zip" or data[:4] == b"PK\x03\x04"


def _from_zip(
    data: bytes, source_name: str, depth: int, result: ExtractResult
) -> list[NormalizedPage]:
    pages: list[NormalizedPage] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = [
            info
            for info in archive.infolist()
            if not info.is_dir()
            and not info.filename.startswith(("__MACOSX", "."))
            and not info.filename.rsplit("/", 1)[-1].startswith(".")
        ][:ZIP_MAX_ENTRIES]

        declared = sum(info.file_size for info in entries)
        if declared > ZIP_MAX_UNCOMPRESSED:
            raise ArchiveTooLargeError(
                f"ZIP {source_name}: uncompressed size {declared} bytes exceeds "
                f"limit of {ZIP_MAX_UNCOMPRESSED} bytes"
            )

        extracted = 0
        for info in entries:
            entry_data = archive.read(info)
            extracted += len(entry_data)
            if extracted > ZIP_MAX_UNCOMPRESSED:
                raise ArchiveTooLargeError(
                    f"ZIP {source_name}: uncompressed size exceeds "
                    f"limit of {ZIP_MAX_UNCOMPRESSED} bytes"
                )
            # this archive is at depth + 1; an entry archive would exceed the
            # nesting limit
            if depth + 1 >= MAX_ARCHIVE_DEPTH and _looks_like_zip(
                info.filename, entry_data
            ):
                logger.warning(
                    "Nested archive %s inside %s skipped (max depth %d)",
                    info.filename,
                    source_name,
                    MAX_ARCHIVE_DEPTH,
                )
                result.nested_archives_skipped += 1
                continue
            try:
                inner = _extract(info.filename, entry_data, depth + 1, result)
            except UnsupportedFormatError:
                continue  # unsupported files inside the archive are skipped
            for page in inner:
                page.page_index = len(pages)
                page.source_name = info.filename
                pages.append(page)
    if not pages:
        skipped = result.nested_archives_skipped
        detail = f" ({skipped} nested archives skipped)" if skipped else ""
        raise UnsupportedFormatError(
            f"ZIP has no supported files: {source_name}{detail}"
        )
    return pages


def _extract(
    filename: str, data: bytes, depth: int, result: ExtractResult
) -> list[NormalizedPage]:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        return _from_pdf(data, filename)
    if ext == "zip":
        return _from_zip(data, filename, depth, result)

    # everything else — try it as an image (Pillow covers jpeg/png/heic/tiff/...),
    # and if that fails, maybe it is a PDF/ZIP without an extension
    for extractor in (_from_pil, _from_pdf):
        try:
            return extractor(data, filename)
        except UnsupportedFormatError:
            raise
        except Exception:
            continue
    if depth < MAX_ARCHIVE_DEPTH:
        try:
            return _from_zip(data, filename, depth, result)
        except (UnsupportedFormatError, ArchiveTooLargeError):
            raise
        except Exception:
            pass
    raise UnsupportedFormatError(f"Unsupported format: {filename}")


def extract_pages(filename: str, data: bytes) -> ExtractResult:
    """Parse a document into normalized pages.

    :raises UnsupportedFormatError: the format is not supported
    :raises ArchiveTooLargeError: uncompressed archive size exceeds the limit
    """
    result = ExtractResult()
    result.pages = _extract(filename, data, 0, result)
    return result
