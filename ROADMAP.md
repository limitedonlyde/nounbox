# Roadmap

Nounbox is self-hosted auto-labeling for object detection: name your classes
in plain English, drop in photos, get boxes from an open-vocabulary model, fix
what is wrong, export YOLO or COCO. Everything runs on your machine.

This roadmap is intentionally short. Dates are deliberately absent — items ship
when they are done, in roughly this order.

## Now

- **Box editing in the review UI** — resize handles, dragging, zoom and pan,
  undo/redo. Reviewing auto-labels means nudging edges far more often than
  deleting boxes, so this comes before anything else.
- **Quality calibration on your own data** — every annotation already records
  which engine produced it, with what confidence, and what the human did with
  it. Turning that into "84% of boxes were accepted untouched; `helmet` is
  corrected most often; suggested threshold 0.37" needs aggregation, not new
  models.
- **Prompt separate from class name** — the label stored in the dataset is
  `forklift`, while the text sent to the model can be `yellow industrial
  forklift`. Wording changes accuracy noticeably.
- **Dry run on a handful of images** — try prompts on four photos in seconds
  instead of discovering the wording was wrong after a batch of five hundred.

## Next

- **Import COCO / YOLO / Pascal VOC** — bring an existing dataset in, review it,
  extend it.
- **Empty frames as first-class data** — mark an image as "checked, nothing
  here". Negative samples reduce false positives when training, and they belong
  in the export.
- **Box prompting** — draw one example, find the rest of that object in the
  image. Useful for things no text prompt describes well: a specific defect, a
  particular part, one SKU among many.
- **Copy annotations from the previous image** — near-free for footage shot from
  a fixed camera.
- **Send images to a remote engine concurrently** — the labeling run posts one
  image at a time, which is right for a CPU engine sharing the worker's own
  cores but leaves a remote GPU almost idle. Measured on a Modal T4: the
  endpoint sustains ~296 images/min at the shipped 4 containers and ~682 at 10,
  while a run through the app gets one image every 1.35 s. The engine is not the
  bottleneck; the loop driving it is.

## Later

- **More detection engines behind the same plugin contract.** A labeler is one
  method, `predict(image, config) -> list[Annotation]`, registered as a Python
  entry point. Adding an engine should never require touching the core.
- **Segmentation masks.** The annotation model already stores polygons; the
  review UI and exporters are what is missing.
- **Fine-tuning loop.** Auto-labeling gets you a first draft; a model trained on
  your corrected data beats zero-shot on your domain. The platform already
  records exactly which boxes a human fixed.

## Not planned

Some things are deliberately out of scope, to keep the core small and
maintainable by a small team:

- Multi-user accounts, roles, task assignment and review queues.
- Training models inside the platform — the export ships a `data.yaml` ready for
  `yolo train`.
- Video-specific tooling, 3D and LiDAR, keypoints and skeletons. CVAT does this
  well and there is no point in a weaker copy.
- A public dataset marketplace.

## Contributing

Ideas and bug reports are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
If you want to add a detection engine, the plugin contract lives in
[`sdk/`](sdk) and needs no changes to the core.
