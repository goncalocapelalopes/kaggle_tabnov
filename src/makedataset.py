import gc
import os
from typing import Tuple

import numpy as np
import pandas as pd
import tqdm

from vars import *

LABEL = "label"
COLS = ["id", "pred"] # some files are weird and have Unnamed: 0 cols

def read_submission_file(fname : str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sep = 20000
    df = pd.read_csv(fname)
    train, test = df.iloc[:sep][COLS], df.iloc[sep:][COLS]
    
    return train, test

def create_datasets() -> None:
    sub_files = os.listdir(SUB_FILES)
    train_all = []
    test_all = []
    for sub_file in tqdm.tqdm(sub_files):
        file_path = os.path.join(SUB_FILES, sub_file)
        train, test = read_submission_file(file_path)
        col = sub_file.split(".")[1]
        train_all.append(pd.DataFrame(data={col: train[FEATURE]}))
        test_all.append(pd.DataFrame(data={col: test[FEATURE]}))

    train_all = pd.concat(train_all, axis=1)
    test_all = pd.concat(test_all, axis=1)
    labels = pd.read_csv(TRAIN_LABELS)
    train_all[LABEL] = labels[LABEL]

    train_all.reset_index().to_feather(os.path.join(PROCESSED, TRAIN_FILE))
    test_all.reset_index().to_feather(os.path.join(PROCESSED, TEST_FILE))
    


def cleanup_processed() -> None:
    for f in os.listdir(PROCESSED):
        os.remove(os.path.join(PROCESSED, f))

if __name__ == "__main__":
    print("Cleaning up processed folder...")
    cleanup_processed()
    print("Creating feather data files...")
    create_datasets()
