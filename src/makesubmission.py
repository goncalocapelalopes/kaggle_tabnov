import os
from typing import Tuple, List

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

from vars import *

def load_test() -> pd.DataFrame:
    data = pd.read_feather(os.path.join(PROCESSED, TEST_FILE))
    data = data.drop(columns="index") # feather index stuff
    data = data.reset_index(drop=True)
    return data

def get_models() -> Tuple(List[LogisticRegression], List[str]):
    client = mlflow.tracking.MlflowClient()
    runs = [
        "2f6bde0fedf2458686797fec9f1afb89",
        "de23810ed7f045648b4b1f7cf3c25143",
        "164369f640884512872b52db2838641a",
        "9a3a94a18d934ea684aff9a81d52f373",
        "778adb810a7b4cde94fc909a8eff5021"
    ]
    
    models = []
    for run in runs:
        models.append(mlflow.sklearn.load_model(client.download_artifacts(run, "model")))
    return models, models[0].feature_names_in_


def predict(data: pd.DataFrame, 
            models: List[LogisticRegression], 
            features: List[str]) -> None:
    X_test = data[features]
    d = {
        "id": np.arange(20000, 40000),
        "pred": np.sum([m.predict_proba(X_test)[:,1] for m in models], axis=0) / len(models)
    }
    assert len(X_test) == len(d["pred"])

    pd.DataFrame(data=d).to_csv("submissions/submission.csv", index=False)
    
if __name__ == "__main__":
    df = load_test()
    models, features = get_models()
    predict(df, models, features)