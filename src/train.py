import os

import mlflow
import mrmr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import pandas as pd

from vars import *

def load_train() -> pd.DataFrame:
    data = pd.read_feather(os.path.join(PROCESSED, TRAIN_FILE))
    data = data.drop(columns="index") # feather index stuff
    data = data.reset_index(drop=True)
    return data

def train(data, top_features=1000) -> None:
    mlflow.set_experiment(LR_EXPERIMENT)
    feature_cols = [c for c in data.columns if c != LABEL]
    X = data[feature_cols]
    y = data[LABEL]
    kf = KFold(shuffle=True, random_state=96)

    mlflow.set_tag("Feature Selector", "mrmr")
    mlflow.log_param("top_features", top_features)
    selected_features = mrmr.mrmr_classif(X, y, K=top_features)
    X = X[selected_features]
    mlflow.log_artifact()

    mlflow.sklearn.autolog()
    losses = []
    for i, (train_index, val_index) in enumerate(kf.split(X)):
        with mlflow.start_run(nested=True):
            print("Fold", i)
            X_train, X_val = X.loc[train_index], X.loc[val_index]
            y_train, y_val = y.loc[train_index], y.loc[val_index]

            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            
            val_preds = clf.predict_proba(X_val)
            loss = log_loss(y_val, val_preds)
            losses.append(loss)
            mlflow.log_metric("fold_val_logloss", loss)
    mlflow.log_metric("val_logloss", sum(losses)/5)
    mlflow.end_run()
    
if __name__ == "__main__":
    train_df = load_train()
    train(train_df)
