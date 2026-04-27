import pandas as pd
import pytest
from mm_eval.data.manifest import validate_manifest

def test_manifest_required_columns_pass():
    df=pd.DataFrame({"image_path":["a.jpg"],"text":["hello"]})
    out=validate_manifest(df,["image_path","text"])
    assert len(out) == 1

def test_manifest_missing_column_raises():
    df=pd.DataFrame({"image_path":["a.jpg"]})
    with pytest.raises(ValueError):
        validate_manifest(df,["image_path","text"])
