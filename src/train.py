import os

import joblib
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, train_test_split
import pandas as pd

from vars import *

def load_train() -> pd.DataFrame:
    data = pd.read_feather(os.path.join(PROCESSED, TRAIN_FILE))
    data = data.drop(columns="index") # feather index stuff
    data = data.reset_index(drop=True)
    return data


def kfold_train(data) -> None:
    mlflow.set_experiment(LR_EXPERIMENT)
    X = data[[c for c in data.columns if c != LABEL]]
    y = data[LABEL]
    kf = KFold(shuffle=True, random_state=96)

    mlflow.sklearn.autolog()
    mlflow.start_run()
    for i, (train_index, val_index) in enumerate(kf.split(X)):
        with mlflow.start_run(nested=True):
            print("Fold", i)
            X_train, X_val = X.loc[train_index], X.loc[val_index]
            y_train, y_val = y.loc[train_index], y.loc[val_index]
            

            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            
            val_preds = clf.predict_proba(X_val)
            loss = log_loss(y_val, val_preds)
            mlflow.log_metric("validation_log_loss", loss)
    mlflow.end_run()

        
    
if __name__ == "__main__":
    train_df = load_train()
    kfold_train(train_df)
