import argparse
import os

import mlflow
import mrmr
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pandas as pd

from vars import *


def load_train() -> pd.DataFrame:
    data = pd.read_feather(os.path.join(PROCESSED, TRAIN_FILE))
    data = data.drop(columns="index") # feather index stuff
    data = data.reset_index(drop=True)
    return data

def train(data: pd.DataFrame, dimred: str, model: str, top_features: int = 1000) -> None:
    mlflow.set_experiment(EXP_MAP[model])
    feature_cols = [c for c in data.columns if c != LABEL]
    X = data[feature_cols]
    y = data[LABEL]

    mlflow.set_tag("Dimensionality Reduction", dimred)
    
    
    if dimred == "mrmr":
        mlflow.log_param("top_features", top_features)
        selected_features = mrmr.mrmr_classif(X, y, K=top_features)
        X = X[selected_features]
    elif dimred == "pca":
        if top_features != "mle":
            top_features = int(top_features)
        dimred = PCA(n_components=top_features)
    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline(steps=[("dimred", dimred), ("logistic", clf)])

    param_grid = {
        "dimred__n_components": [50, 100, 300, 500],
        "logistic__C": np.linspace(0.1, 1, num=4)
        
    }
    mlflow.sklearn.autolog()
    search = GridSearchCV(pipe, param_grid, n_jobs=-1, scoring="neg_log_loss")
    search.fit(X, y)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--dimred", type=str, choices=["pca", "mrmr"], default="mrmr")
    parser.add_argument("--model", type=str, choices=["lr"], default="lr")
    parser.add_argument("--topfeatures", type=str, default="1000")
    args = parser.parse_args()
    
    dimred = args.dimred
    model = args.model
    topfeatures = args.topfeatures
    train_df = load_train()
    train(train_df, dimred, model, top_features=topfeatures)
