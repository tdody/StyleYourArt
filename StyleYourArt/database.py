"""
Create connections to database
"""
import os, sys
sys.path.append('..')
from pymongo import MongoClient
from StyleYourArt import dataloader
from StyleYourArt import secrets as sec
import pandas as pd
import json
import requests
import math

class DB:
    '''
    Database class
    '''
    def __init__(self):
        '''
        Creates database engine
        '''
        self.client = MongoClient(
            "mongodb+srv://%s:%s@%s.mongodb.net/test?retryWrites=true&w=majority" % (sec.mongouser,
                                                                                     sec.mongopwd,
                                                                                     sec.mongohost),
            connect=False)
        
        ## database
        self.db = self.client.ArtStyles

        ## collections
        self.paintings = self.db.paintings

    def update_paintings(self):
        '''
        Update content of database based on content of csv file.
        '''

        ## load csv
        df_paintings = dataloader.load_clean_data('../data/csv', '../data/images/', verbose=True)
        records = df_paintings.to_dict('records')

        ## eliminate records without image
        df_paintings['valid_url'] = df_paintings['image'].map(lambda x: db.is_url_image)
        df_paintings = df_paintings[df_paintings['valid_url']]

        ## clear existing content
        self.paintings.delete_many({})

        ## update database
        self.paintings.insert_many(records)
        print("... database updated")

    def get_painting_for_style(self, stylename, count):
        '''
        Return a random subset of painting record based on a provided style name.
        
        Arguments:
            stylename: string, style name
            count: integer, equal to the number of records to be retrieved
        '''
        ## pipeline
        pipeline = [
            {
                '$project': {
                    'artist_name':1,
                    'completion_year':1,
                    'style':1,
                    'title':1,
                    'image':1,
                    'image_exists':1,
                    'valid_url':1
                }
            },
            {
                '$match' : {
                    'style': stylename,
                    'image_exists': True,
                    'valid_url': 1.0
                    }
            },
            {
                '$sample': {
                    'size': count
                }
            }
        ]
        
        ## fetch
        results = list(self.paintings.aggregate(pipeline))
        return results

    def is_url_image(self, image_url):
        image_formats = ("image/png", "image/jpeg", "image/jpg")
        r = requests.head(image_url)
        if r.headers["content-type"] in image_formats:
            return True
        return False

if __name__=="__main__":

    my_DB = DB()

    my_DB.get_painting_for_style("Baroque", 10)