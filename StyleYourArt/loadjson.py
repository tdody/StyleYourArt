#!/usr/bin/env python

"""
Module used to compile the json files into a csv.
"""

import os
import pandas as pd
import json
import csv
from glob import glob

MAIN_CSV = "./data/csv/"
LOG = "./data/csv/done.csv"
MASTER = "./data/csv/master.csv"

MASK = ['contentId', 'title', 'artistName', 'completitionYear', 'image', 'artistUrl', 'url', 'height', 'width', 'style']

def merge_json(datadir):

    ## read done
    #with open(LOG, newline='') as f:
    #    reader = csv.reader(f)
    #    done = list(reader)
    with open(LOG, 'r') as f:
        done = f.read().splitlines()

    for f in sorted(glob(datadir + '/*.json')):
        ## check if not in done
        if not f in done:

            ## read json file
            df = pd.read_json(f)

            ## skip empty json
            if df.shape[0] == 0: continue

            ## filter using selected columns
            df = df[MASK]

            ## add formatted painter
            df['json'] = os.path.split(f)[-1]

            ## save df to master csv
            df.to_csv(MASTER, mode='a', header=False)

            ## add json file to list of covered files
            with open(LOG , 'a') as fd:
                wr = csv.writer(fd, dialect='excel')
                wr.writerow([f])

if __name__ == "__main__":
    print(merge_json('./data/meta'))