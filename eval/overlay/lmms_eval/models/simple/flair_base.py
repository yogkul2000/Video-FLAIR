from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.flair import Flair


@register_model("flair_base")
class FlairBase(Flair):
    def __init__(self, pretrained: str, mode: str = "NONE", timestamps_line: bool = False, **kwargs) -> None:
        super().__init__(pretrained=pretrained, mode=mode, timestamps_line=timestamps_line, **kwargs)
