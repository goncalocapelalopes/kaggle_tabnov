import os

import joblib
import numpy as np
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


def kfold_train(data) -> None:
    estimators = []
    train_scores = []
    test_scores = []
    X = data[[c for c in data.columns if c != LABEL]]
    y = data[LABEL]
    kf = KFold(shuffle=True, random_state=96)

    i = 0
    for train_index, test_index in kf.split(X):
        print(i)
        i += 1
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        train_preds = clf.predict_proba(X_train)
        test_preds = clf.predict_proba(X_test)
    
        estimators.append(clf)
        train_scores.append(log_loss(y_train, train_preds))
        test_scores.append(log_loss(y_test, test_preds))

    print(train_scores)
    print(test_scores)
    best_estimator = estimators[test_scores.index(max(test_scores))]
    joblib.dump(best_estimator, os.path.join(MODELS, "lr.joblib"))
    

if __name__ == "__main__":
    train_df = load_train()
    kfold_train(train_df)
