import sys
from types import SimpleNamespace as NS
import pandas as pd
import pytest
from PIL import Image
from videolens.models import vision

def test_frame_aggregation_and_missing_frames(tmp_path, monkeypatch):
    folder = tmp_path / "video1"
    folder.mkdir()
    for i in range(4):
        Image.new("RGB", (8,8), "red").save(folder / f"{i}.png")
    calls = []
    def detector(image, candidate_labels):
        calls.append(image.size)
        return [{"label": name, "score": .8 if i == 2 else .05} for i,name in enumerate(candidate_labels)]
    monkeypatch.setitem(sys.modules, "transformers", NS(pipeline=lambda **kwargs:detector))
    df = pd.DataFrame({"video_id":["video1", "no_frames"]})
    result = vision(df, tmp_path)
    assert len(calls) == 3
    assert len(result) == 1
    assert result.iloc[0].vision_category == "food"
    assert result.iloc[0].frame_count == 3

@pytest.mark.parametrize("video_id", ["../escape", "..", "a/b"])
def test_frame_path_escape_rejected(tmp_path, monkeypatch, video_id):
    monkeypatch.setitem(sys.modules, "transformers", NS(pipeline=lambda **kwargs:None))
    with pytest.raises(ValueError):
        vision(pd.DataFrame({"video_id":[video_id]}), tmp_path)
