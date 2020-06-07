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

        ## load csv
        df_paintings = dataloader.load_clean_data('../data/csv', '../data/images/', verbose=True)
        records = df_paintings.to_dict('records')

        ## clear existing content
        self.paintings.delete_many({})

        ## update database
        self.paintings.insert_many(records)
        print("... database updated")

    def get_painting_for_style(self, stylename, count):

        ## pipeline
        pipeline = [
            {
                '$project': {
                    'artist_name':1,
                    'completion_year':1,
                    'style':1,
                    'title':1,
                    'image':1,
                    'image_exists':1
                }
            },
            {
                '$match' : {
                    'style': stylename,
                    'image_exists': True
                    }
            },
            {
                '$sample': {
                    'size': count
                }
            }
        ]
        i = 0
        return self.paintings.aggregate(pipeline)

if __name__=="__main__":

    my_DB = DB()

    my_DB.get_painting_for_style("Baroque", 10)