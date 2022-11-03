import os
from typing import Tuple, List

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd

from vars import *

def load_test() -> pd.DataFrame:
    data = pd.read_feather(os.path.join(PROCESSED, TEST_FILE))
    data = data.drop(columns="index") # feather index stuff
    data = data.reset_index(drop=True)
    return data

def get_models() -> Pipeline:
    client = mlflow.tracking.MlflowClient()

    run = "cf1ec42bc1bc43db901711fd29c69b53"
    model = mlflow.sklearn.load_model(client.download_artifacts(run, "best_estimator"))
    return model


def predict(data: pd.DataFrame, 
            model: List[LogisticRegression]) -> None:
    X_test = data
    d = {
        "id": np.arange(20000, 40000),
        "pred": model.predict_proba(X_test)[:,1]
    }
    assert len(X_test) == len(d["pred"])

    pd.DataFrame(data=d).to_csv("submissions/submission.csv", index=False)
    
if __name__ == "__main__":
    df = load_test()
    model = get_models()
    predict(df, model)