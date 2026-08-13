# autolabelui-sdk

The labeler plugin contract for AutoLabelUi. Implement the `Labeler` protocol and
register it through entry points — the platform picks the engine up automatically.

```python
from autolabelui_sdk import Annotation, BBox, Capability

class MyLabeler:
    name = "my-ocr"
    version = "0.1.0"
    capabilities = {Capability.DETECTION, Capability.RECOGNITION}

    def predict(self, image: bytes, config: dict) -> list[Annotation]:
        return [Annotation(geometry=BBox(10, 10, 100, 30), text="hello", confidence=0.95)]
```

```toml
# pyproject.toml of your plugin
[project.entry-points."autolabelui.labelers"]
my_ocr = "my_package:MyLabeler"
```
