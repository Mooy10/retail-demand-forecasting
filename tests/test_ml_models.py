import numpy as np
import pandas as pd

from src.ml_models import fit_model


def test_ml_pipeline_predicts_known_categories_and_clips_nonnegative():
    X = pd.DataFrame({"store_id": ["A", "A", "B", "B"], "dept_id": ["D1", "D2", "D1", "D2"], "horizon": [1, 2, 1, 2], "lag_1": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([2.0, 3.0, 4.0, 5.0])
    model = fit_model("hist_gradient_boosting", X, y, ["store_id", "dept_id"], ["horizon", "lag_1"], params={"max_iter": 5, "random_state": 42})
    pred = model.predict(X)
    assert len(pred) == len(X)
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()