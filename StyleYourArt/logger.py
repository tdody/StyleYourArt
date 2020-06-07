#!/usr/bin/env python
"""
module with functions to enable logging
"""

import time
from time import strftime 
import os
import re
import csv
import sys
import uuid
import joblib
from datetime import date, datetime
import pandas as pd

def update_train_log(MODEL_VERSION, MODEL_BASE_TAG, MODEL_VERSION_NOTE, train_shape, test_shape, train_score, test_score,
                    EPOCHS, BATCH_SIZE, callbacks, runtime, optimizer_config, test=False, from_notebook=False):
    """
    update training log

    To be saved:
        - model version
        - base model tag ResNet50, VGG16
        - model version notes
        - input_shape
        - test_shape
        - train_score (dictionary): accuracy, F1-score
        - test_score (dictionary): accuracy, F1-score
        - is test? (i.e is the full dataset used?)
        - n_epochs
        - batch_size
        - callbacks
        - optimizer
        - runtime
        - optimizer_config
    """

    if from_notebook:
        relative = ".."
    else:
        relative = "."

    ## define cyclic name for log
    today = date.today()

    ## create path for logs if needed
    if not os.path.exists(os.path.join(relative,"logs")):
        os.mkdir(os.path.join(relative,"logs"))
    if test:
        logfile = os.path.join(relative, "logs", "train-test.log")
    else:
        logfile = os.path.join(relative, "logs", "train-{}-{}.log".format(today.year, today.month))

    if not os.path.exists(os.path.join(relative, "logs")):
        os.mkdir(os.path.join(relative, "logs"))

    ## write the data to csv file
    header = [
        "unique_id", "timestamp", "model_version", "base_model", "x_train_shape",
        "x_test_shape", "train_score", "test_score", "n_epochs",
        "batch_size", "callbacks", "optimizer", "optimizer_config", "runtime",
        "model_version_note"]
    
    ## determine if header needs to be written
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    
    ## update log
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)
        to_write = map(str,[
            uuid.uuid4(),
            strftime("%Y_%m_%_d_%H:%M:%S"),
            MODEL_VERSION, MODEL_BASE_TAG, train_shape,
            test_shape, train_score,
            test_score, EPOCHS,
            BATCH_SIZE, callbacks,
            optimizer_config,
            runtime])
        writer.writerow(to_write)


def update_predict_log(y_pred,query,runtime,MODEL_VERSION,test=False):
    """
    update predict log file
    """

    ## define cyclic name for log
    today = date.today()
    if test:
        logfile = os.path.join("logs", "predict-test.log")
    else:
        logfile = os.path.join("logs", "predict-{}-{}.log".format(today.year, today.month))

    ## write the data to a csv file    
    header = ['unique_id','timestamp','y_pred','query','model_version','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(),y_pred,query,
                            MODEL_VERSION,runtime])
        writer.writerow(to_write)


def find_latest_predict_log(train=True):
    """
    Fetch latest log and return it into a formatted pandas dataframe
    """

    ## main log locations    
    log_dir = './logs'
    
    ## set prefix train or predict
    if train:
        prefix = "train-test"
    else:
        prefix = "predict"

    ## find all relevant logs
    logs = [f for f in os.listdir(log_dir) if re.search(prefix,f)]

    if len(logs)==0:
        return None
    else:
        ## find most recent
        logs.sort()
        df = pd.read_csv(os.path.join(log_dir,logs[-1]))

        if train:
            ## filter columns
            columns = ['timestamp', 'tag','start_date', 'end_date', 'x_shape', 'eval_test', 'model_version', 'runtime']

        else:
            ## filter columns
            columns = ['timestamp', 'y_pred', 'query', 'model_version', 'runtime']

        ## format time stamp
        df['timestamp'] = pd.Series([datetime.fromtimestamp(x) for x in df['timestamp']]).dt.strftime('%m-%d-%Y %r')

        return df[columns]


if __name__ == "__main__":

    """
    basic test procedure for logger.py
    """

    from models import MODEL_VERSION, MODEL_VERSION_NOTE
    
    ## train logger
    MODEL_VERSION = "a"
    MODEL_BASE_TAG = "b"
    MODEL_VERSION_NOTE = "c"
    train_shape = "d"
    test_shape = "e"
    train_score = "f"
    test_score = "g"
    EPOCHS = "h"
    BATCH_SIZE = "i"
    callbacks = "j"
    runtime = "k"
    optimizer_config = "l"
    test=False
    from_notebook=False
    update_train_log(MODEL_VERSION, MODEL_BASE_TAG, MODEL_VERSION_NOTE, train_shape, test_shape, train_score, test_score,
                    EPOCHS, BATCH_SIZE, callbacks, runtime, optimizer_config, test, from_notebook)
    
    ## predict logger
    #update_predict_log("[0]","[0.6,0.4]","['united_states', 24, 'aavail_basic', 8]",
    #                   "00:00:01",MODEL_VERSION, test=True)