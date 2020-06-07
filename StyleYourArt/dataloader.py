#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module used to load our data.

Assumptions:
    1. The *.json files have been loaded using the wikiart module.
    2. Using loadjson.py, the json files have been merged into a csv file.
"""

import os
import re
import time
import numpy as np
import pandas as pd
import json
import shutil
import csv
from collections import defaultdict

def load_data(data_dir,image_dir,verbose=True):
    """
    Load painting information stored in specified location.
    
    Arguments:
        - data_dir (string): location of csv/json files
    
    Outputs:
        - pandas dataframe created from the content of the csv file.
    """
    ## check if file exists
    if not os.path.exists(data_dir):
        raise Exception("specified data directory does not exist.")

    ## create full directory
    full_path = os.path.join(data_dir, 'master.csv')

    ## check if full path exists
    if not os.path.isfile(full_path):
        raise Exception("master.csv not in specified directory.")

    ## load data
    if verbose: print("... loading painting data from master.csv")
    df = pd.read_csv(full_path, index_col=0, encoding='utf-8')
    if verbose: print("... painting data imported from master.csv")

    ## column names to be used
    clean_columns = ['content_id', 'title', 'artist_name', 'completion_year', 
                     'image', 'artist_url', 'url', 'height', 'width', 'style', 'json_file']
    clean_columns = sorted(clean_columns)

    ## clean column names
    ## contentId -> content_id
    ## artistName -> artist_name
    ## completitionYear -> completion_year
    ## artistUrl -> artist_url
    ## json -> json_file
    if verbose: print("... setting dataframe schema")
    cols = df.columns.tolist()
    if "contentId" in cols:
        df.rename(columns={"contentId": "content_id"}, inplace=True)
    if "completitionYear" in cols:
        df.rename(columns={"completitionYear": "completion_year"}, inplace=True)
    if "artistName" in cols:
        df.rename(columns={"artistName": "artist_name"}, inplace=True)
    if "artistUrl" in cols:
        df.rename(columns={"artistUrl": "artist_url"}, inplace=True)
    if "json" in cols:
        df.rename(columns={"json": "json_file"}, inplace=True)
    
    ## verify column compatibility
    df = df[clean_columns]
    if sorted(df.columns.tolist()) != clean_columns:
        raise Exception("column names do not match schema")

    ## check if full path exists
    if not os.path.exists(image_dir):
        raise Exception("specified image dir does not exist")

    ## create image location
    if image_dir[-1]=="/":
        image_dir = image_dir[0:-1]

    ## remove unnecessary information
    #df['format'] = df['image'].str.split('.').str[-1]
    #df.loc[df['format']=='jpeg', 'format'] = 'jpg'
    df['format'] = 'jpg'
    df['file_loc'] = image_dir + "/" + df['artist_url'] + "/" + df['completion_year'].map(lambda x: str(int(x)) if pd.notnull(x) else "unknown-year") + "/" + df['content_id'].astype(str) + '.'+ df['format']

    ## remove record without image
    df['image_exists'] = df['file_loc'].apply(lambda x: os.path.exists(x))
    df = df[df['image_exists']]

    ## save all styles
    if verbose: print("... exporting unique styles to styles.csv")
    pd.Series(df['style'].unique()).to_csv(os.path.join(data_dir, 'styles.csv'))
    if verbose: print("... data loaded")
    return df

def load_clean_data(data_dir,image_dir,verbose=True):
    """
    Read data from clean csv and filter missing content and top 25 styles.
    Return a pandas dataframe.
    """
    ## extract data from csv file
    df = load_data(data_dir, image_dir, verbose=verbose)

    ## find most common styles
    if verbose: print("... finding top 25 most popular styles")
    top_25_styles = df['style'].value_counts().head(25) / df.shape[0] * 100.
    top_25_styles = top_25_styles.index.to_list()

    ## filter dataset using selected 25 styles
    if verbose:
        print("... filtering data.")
        print("   ... record count before filtering: {}".format(df.shape[0]))
    df = df[df['style'].isin(top_25_styles)]
    if verbose: print("   ... record count after filtering: {}".format(df.shape[0]))

    ## group styles
    Renaissance = [
        'Early Renaissance', 'Northern Renaissance', 
        'High Renaissance', 'Mannerism (Late Renaissance)'
    ]
    Baroque = ['Rococo']
    Impressionism = ['Post-Impressionism']
    Abstract = ['Abstract Expressionism', 'Art Informel']

    for val in Renaissance:
        df['style'] = df['style'].replace(val, 'Renaissance')
    for val in Baroque:
        df['style'] = df['style'].replace(val, 'Baroque')
    for val in Impressionism:
        df['style'] = df['style'].replace(val, 'Impressionism')
    for val in Abstract:
        df['style'] = df['style'].replace(val, 'Abstract Art')

    ## Rename missing artist names
    if verbose: print("... replace missing values")
    df['artist_name'] = df['artist_name'].str.replace("￿", "UNKNOWN")
    return df

if __name__ == "__main__":

    ## load data
    df = load_clean_data('./data/csv', './data/images/', verbose=True)
    print("SIZE: ", df.shape)
    print("STYLE COUNT: ", df['style'].unique().shape)
    print("STYLES: ", df['style'].value_counts())