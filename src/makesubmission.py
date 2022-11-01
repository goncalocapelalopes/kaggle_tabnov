import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import pandas as pd

from vars import *

def load_test() -> pd.DataFrame:
    data = pd.read_feather(os.path.join(PROCESSED, TEST_FILE))
    data = data.drop(columns="index") # feather index stuff
    data = data.reset_index(drop=True)
    return data

def predict(data) -> None:
    clf = joblib.load(LR_MODEL)
    X_test = data
    d = {
        "ids": np.arange(20000, 40000),
        "pred": clf.predict_proba(X_test)[:, 1]
    }
    print(len(X_test))
    print(len(d["pred"]))
    pd.DataFrame(data=d).to_csv("submission.csv", index=False)
    
if __name__ == "__main__":
    df = load_test()
    predict(df)