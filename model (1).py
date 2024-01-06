import pandas as pd
import numpy as np
from pathlib import Path
from data import popularity_data, collaborative_data
import implicit
import scipy
import joblib
from glob import glob
import os

#model class for popularity recommender
class Popularity:
    def __init__(self, model_directory = 'models'):
        self._popularity_recommender = None
        self.model_directory = model_directory
     #create dataset fo the model   
    def create(self):
        self.data = None
        
        store = popularity_data()
        df1 = store.get_user_artist_data('./dataset/user_artists.dat')
        df2 = store.get_artist_data('./dataset/artists.dat')
        storedata = store.get_combined_dataset(df1, df2)
        self.data = storedata.groupby(['artistName']).agg({'weight':'count'}).reset_index()
        self.data.sort_values(['weight', 'artistName'], ascending=False, inplace=True)
        self.data['Rank'] = self.data['weight'].rank(ascending=False)
        self.data.rename(columns = {'weight': 'listening_count'}, inplace=True)
        self.data.reset_index(drop= True, inplace= True)
        self.data.drop(columns='Rank', inplace=True)
        self.data.drop(columns='listening_count', inplace=True)
        self._popularity_recommender = self.data.head(10).to_dict()
        
    #perform recommendations    
    def recommend(self, user_id):
        user_recommendations = self._popularity_recommender
        return user_recommendations
    
    
    def dump(self):
        #create model filepath
        filepath = os.path.join(self.model_directory, ('popularity_based_recommender.pkl'))
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))   
        #save model
        joblib.dump(self._popularity_recommender, filepath)
        return filepath
     
    def load(self):
        model_path = os.path.join(self.model_directory, ('popularity_based_recommender.pkl'))
        self.model = joblib.load(model_path)
        return self.model
        
        
    
#model class for collaborative filtering   
class Implicit_recommender:
    
    def __init__(
        self,
        artist_retriever: collaborative_data(),
        implicit_model: implicit.recommender_base.RecommenderBase,
        model_directory = 'models2'
    ):
        
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model
        self.model_directory = model_directory
        
    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix):
        self.implicit_model.fit(user_artists_matrix)
        
    def recommend(
        self,
        user_id,
        user_artists_matrix: scipy.sparse.csr_matrix,
        n: int = 10):
        artists_ids, scores = self.implicit_model.recommend(user_id, user_artists_matrix[n])
        artists = [ self.artist_retriever.get_artist_name(artist_id)
                   for artist_id in artists_ids]
        return artists, scores
    def dump(self):
        filepath = os.path.join(self.model_directory, ('implicit.pkl'))
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        joblib.dump(self.implicit_model, filepath)
        return filepath
    def load(self):
        model_path = os.path.join(self.model_directory, ('implicit.pkl'))
        self.model = joblib.load(model_path)
        return self.model
    

                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                       
                        
         