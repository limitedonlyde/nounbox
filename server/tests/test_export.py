"""Экспорт датасета: нормировка YOLO, детерминированный сплит, COCO instances."""

import io
import json
import uuid
import zipfile

import pytest
from PIL import Image as PILImage

from app.models import Annotation, AnnotationStatus, Image
from app.services import export as export_service


def png_bytes(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def make_image(
    width: int = 640, height: int = 480, image_id: uuid.UUID | None = None
) -> Image:
    return Image(
        id=image_id or uuid.uuid4(),
        document_id=uuid.uuid4(),
        page_index=0,
        s3_key="images/x.png",
        width=width,
        height=height,
    )


def bbox_ann(
    label: str, x: float, y: float, w: float, h: float, text: str | None = None
):
    return Annotation(
        id=uuid.uuid4(),
        image_id=uuid.uuid4(),
        geometry={"type": "bbox", "x": x, "y": y, "width": w, "height": h},
        label=label,
        text=text,
        attrs={},
        confidence=0.87,
        source={"type": "engine", "name": "owlv2"},
        status=AnnotationStatus.ACCEPTED,
    )


def polygon_ann(label: str, points: list[list[float]], text: str | None = None):
    return Annotation(
        id=uuid.uuid4(),
        image_id=uuid.uuid4(),
        geometry={"type": "polygon", "points": points},
        label=label,
        text=text,
        attrs={},
        confidence=0.5,
        source={"type": "engine", "name": "rapidocr"},
        status=AnnotationStatus.EDITED,
    )


def item(image: Image, annotations: list[Annotation]) -> export_service.ExportItem:
    data = png_bytes(image.width, image.height)
    return export_service.ExportItem(image, data, annotations)


def read_zip(data: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(data))


def yolo_label(zf: zipfile.ZipFile, image: Image) -> str:
    """Содержимое labels/<split>/<stem>.txt для изображения."""
    stem = image.id.hex
    return zf.read(f"labels/{export_service._split_of(stem)}/{stem}.txt").decode()


def build(fmt: str, items, task_type: str = "detection", classes=("carpet", "sofa")):
    return read_zip(
        export_service.build_zip(fmt, items, "proj-1", task_type, classes)
    )


# --- список форматов ---
def test_formats_filtered_by_task_type():
    assert export_service.formats_for("detection") == ("yolo_detect", "coco")
    ocr = ("paddleocr_det", "paddleocr_rec", "coco")
    assert export_service.formats_for("ocr") == ocr
    # пустой/неизвестный task_type — дефолт detection
    assert export_service.formats_for(None) == ("yolo_detect", "coco")
    assert export_service.formats_for("something_else") == ("yolo_detect", "coco")


def test_ocr_format_rejected_for_detection_project():
    items = [item(make_image(), [bbox_ann("carpet", 0, 0, 10, 10)])]
    with pytest.raises(export_service.ExportError, match="task_type=detection"):
        export_service.build_zip("paddleocr_det", items, "p", "detection", ("carpet",))


def test_yolo_rejected_for_ocr_project():
    items = [item(make_image(), [bbox_ann("text_line", 0, 0, 10, 10, text="hi")])]
    with pytest.raises(export_service.ExportError, match="task_type=ocr"):
        export_service.build_zip("yolo_detect", items, "p", "ocr", ())


def test_unknown_format():
    items = [item(make_image(), [bbox_ann("carpet", 0, 0, 10, 10)])]
    with pytest.raises(export_service.ExportError, match="Unknown format"):
        export_service.build_zip("voc", items, "p", "detection", ("carpet",))


def test_yolo_without_classes_fails():
    items = [item(make_image(), [bbox_ann("carpet", 0, 0, 10, 10)])]
    with pytest.raises(export_service.ExportError, match="no classes"):
        export_service.build_zip("yolo_detect", items, "p", "detection", ())


# --- YOLO ---
def test_yolo_normalization_known_numbers():
    image = make_image(640, 480)
    # bbox 100,50 200x100 → cx=200/640=0.3125, cy=100/480=0.208333, w=0.3125, h=0.208333
    zf = build("yolo_detect", [item(image, [bbox_ann("sofa", 100, 50, 200, 100)])])
    stem = image.id.hex
    split = export_service._split_of(stem)
    label = zf.read(f"labels/{split}/{stem}.txt").decode()
    assert label == "1 0.312500 0.208333 0.312500 0.208333"
    assert f"images/{split}/{stem}.png" in zf.namelist()


def test_yolo_class_index_follows_sort_order():
    image = make_image(100, 100)
    annotations = [
        bbox_ann("chandelier", 0, 0, 50, 50),
        bbox_ann("carpet", 50, 50, 50, 50),
    ]
    zf = build(
        "yolo_detect",
        [item(image, annotations)],
        classes=("carpet", "sofa", "chandelier"),
    )
    lines = yolo_label(zf, image).split("\n")
    assert [line.split()[0] for line in lines] == ["2", "0"]
    data_yaml = zf.read("data.yaml").decode()
    assert "nc: 3" in data_yaml
    assert "train: images/train\nval: images/val" in data_yaml
    assert '  0: "carpet"\n  1: "sofa"\n  2: "chandelier"' in data_yaml
    # path не задаём — иначе ultralytics ищет датасет относительно cwd
    assert "path:" not in data_yaml


def test_yolo_clips_box_to_image():
    image = make_image(100, 100)
    # бокс вылезает за левый верхний угол: −10,−10 30x30 → 0,0 20x20
    zf = build("yolo_detect", [item(image, [bbox_ann("carpet", -10, -10, 30, 30)])])
    assert yolo_label(zf, image) == "0 0.100000 0.100000 0.200000 0.200000"


def test_yolo_drops_degenerate_box_but_keeps_image():
    image = make_image(100, 100)
    annotations = [
        bbox_ann("carpet", 200, 200, 10, 10),  # целиком за кадром
        bbox_ann("carpet", 10, 10, 20, 20),
    ]
    zf = build("yolo_detect", [item(image, annotations)])
    lines = yolo_label(zf, image).split("\n")
    assert lines == ["0 0.200000 0.200000 0.200000 0.200000"]
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["annotations"] == 1
    assert manifest["images"] == 1


def test_yolo_polygon_becomes_bounding_box():
    image = make_image(200, 100)
    polygon = polygon_ann("carpet", [[10, 20], [110, 20], [110, 70], [10, 70]])
    zf = build("yolo_detect", [item(image, [polygon])])
    # x 10..110 из 200 → cx=0.3, w=0.5; y 20..70 из 100 → cy=0.45, h=0.5
    assert yolo_label(zf, image) == "0 0.300000 0.450000 0.500000 0.500000"


def test_yolo_skips_annotations_with_unknown_label():
    image = make_image(100, 100)
    annotations = [
        bbox_ann("carpet", 0, 0, 50, 50),
        bbox_ann("lamp", 50, 50, 50, 50),  # класс удалён из проекта
    ]
    zf = build("yolo_detect", [item(image, annotations)])
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["annotations"] == 1
    assert manifest["skipped_labels"] == {"lamp": 1}
    assert manifest["classes"] == ["carpet", "sofa"]
    assert manifest["task_type"] == "detection"


# --- детерминированность сплита ---
def _splits(zf: zipfile.ZipFile) -> dict[str, str]:
    out = {}
    for name in zf.namelist():
        if name.startswith("labels/"):
            _, split, filename = name.split("/")
            out[filename.removesuffix(".txt")] = split
    return out


def _many_items(count: int) -> list[export_service.ExportItem]:
    return [
        item(
            make_image(100, 100, uuid.UUID(int=i)),
            [bbox_ann("carpet", 10, 10, 20, 20)],
        )
        for i in range(count)
    ]


def test_split_is_deterministic_between_exports():
    items = _many_items(30)
    first = _splits(build("yolo_detect", items))
    second = _splits(build("yolo_detect", items))
    assert first == second
    assert set(first.values()) == {"train", "val"}  # обе части непустые


def test_split_survives_added_and_removed_images():
    items = _many_items(30)
    full = _splits(build("yolo_detect", items))
    without_last = _splits(build("yolo_detect", items[:-1]))
    # у оставшихся снимков сплит не поехал — имя файла не зависит от позиции
    assert all(full[stem] == split for stem, split in without_last.items())
    assert len(without_last) == len(full) - 1


def test_both_splits_are_non_empty_on_small_datasets():
    # ultralytics падает, если папка val пустая, а на 2-5 снимках хеш легко
    # может отправить всех в train
    for count in range(1, 8):
        names = build("yolo_detect", _many_items(count)).namelist()
        assert any(n.startswith("images/train/") for n in names), count
        assert any(n.startswith("images/val/") for n in names), count
        assert any(n.startswith("labels/val/") for n in names), count


def test_small_dataset_split_is_still_deterministic():
    items = _many_items(3)  # все три хешируются в train, один уезжает в val
    assert _splits(build("yolo_detect", items)) == _splits(build("yolo_detect", items))


def test_split_ratio_is_roughly_80_20():
    splits = _splits(build("yolo_detect", _many_items(400)))
    val = sum(1 for s in splits.values() if s == "val")
    assert 0.12 < val / len(splits) < 0.28


# --- COCO ---
def test_coco_detection_instances():
    image = make_image(640, 480)
    annotations = [
        bbox_ann("chandelier", 10, 20, 30, 40),
        polygon_ann("carpet", [[10, 20], [110, 20], [110, 70], [10, 70]]),
    ]
    zf = build(
        "coco", [item(image, annotations)], classes=("carpet", "sofa", "chandelier")
    )
    coco = json.loads(zf.read("annotations.json"))

    # категории — классы проекта в порядке sort_order, id 1-based
    assert coco["categories"] == [
        {"id": 1, "name": "carpet"},
        {"id": 2, "name": "sofa"},
        {"id": 3, "name": "chandelier"},
    ]
    assert coco["images"] == [
        {"id": 1, "file_name": "images/00001.png", "width": 640, "height": 480}
    ]

    first, second = coco["annotations"]
    assert first["category_id"] == 3
    assert first["bbox"] == [10, 20, 30, 40]
    assert first["area"] == 1200
    assert first["iscrowd"] == 0
    assert first["segmentation"] == []  # только боксы, сегментации в проекте нет
    assert "text" not in first["attributes"]
    # полигон приведён к ограничивающему прямоугольнику
    assert second["category_id"] == 1
    assert second["bbox"] == [10, 20, 100, 50]
    assert second["area"] == 5000
    assert "images/00001.png" in zf.namelist()


def test_coco_detection_skips_unknown_label():
    image = make_image(100, 100)
    annotations = [bbox_ann("carpet", 0, 0, 10, 10), bbox_ann("lamp", 0, 0, 10, 10)]
    zf = build("coco", [item(image, annotations)], classes=("carpet",))
    coco = json.loads(zf.read("annotations.json"))
    assert [a["category_id"] for a in coco["annotations"]] == [1]
    assert json.loads(zf.read("manifest.json"))["skipped_labels"] == {"lamp": 1}


def test_coco_ocr_keeps_text_and_segmentation():
    image = make_image(200, 100)
    points = [[10, 20], [110, 20], [110, 70], [10, 70]]
    zf = build(
        "coco",
        [item(image, [polygon_ann("text_line", points, "Итого")])],
        task_type="ocr",
        classes=(),
    )
    coco = json.loads(zf.read("annotations.json"))
    assert coco["categories"] == [{"id": 1, "name": "text_line"}]
    entry = coco["annotations"][0]
    assert entry["attributes"]["text"] == "Итого"
    assert entry["segmentation"] == [[10, 20, 110, 20, 110, 70, 10, 70]]
    assert entry["bbox"] == [10, 20, 100, 50]


# --- ocr-форматы не сломались ---
def test_paddleocr_det_unchanged():
    image = make_image(200, 100)
    annotations = [
        polygon_ann("text_line", [[10, 20], [110, 20], [110, 70], [10, 70]], "Итого"),
        bbox_ann("text_line", 0, 0, 10, 10, text="1\t2\n3"),
    ]
    zf = build("paddleocr_det", [item(image, annotations)], task_type="ocr", classes=())
    line = zf.read("label.txt").decode()
    path, payload = line.split("\t", 1)
    assert path == "images/00001.png"
    entries = json.loads(payload)
    assert entries[0] == {
        "transcription": "Итого",
        "points": [[10, 20], [110, 20], [110, 70], [10, 70]],
    }
    assert entries[1]["points"] == [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert "images/00001.png" in zf.namelist()


def test_paddleocr_rec_crops():
    image = make_image(200, 100)
    annotations = [
        bbox_ann("text_line", 10, 10, 50, 20, text="Итого\t7"),
        bbox_ann("text_line", 10, 10, 50, 20),  # без текста — в rec не попадает
    ]
    zf = build("paddleocr_rec", [item(image, annotations)], task_type="ocr", classes=())
    lines = zf.read("label.txt").decode().split("\n")
    assert lines == ["crops/00001_000.png\tИтого 7"]
    assert "crops/00001_000.png" in zf.namelist()


def test_manifest_lists_classes_and_task_type():
    image = make_image(100, 100)
    zf = build("yolo_detect", [item(image, [bbox_ann("carpet", 0, 0, 50, 50)])])
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["format"] == "yolo_detect"
    assert manifest["task_type"] == "detection"
    assert manifest["classes"] == ["carpet", "sofa"]
    assert manifest["statuses_included"] == ["accepted", "edited"]
    assert manifest["project_id"] == "proj-1"
    # единственный снимок идёт и в train, и в val: пустой val ломает ultralytics
    assert (manifest["train_images"], manifest["val_images"]) == (1, 1)


def test_empty_export_fails():
    with pytest.raises(export_service.ExportError, match="No verified annotations"):
        export_service.build_zip("yolo_detect", [], "p", "detection", ("carpet",))
