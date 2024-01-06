import pandas as pd
from pathlib import Path
from scipy.sparse import coo_matrix

#class for preparing data for popularity recommender model
class popularity_data:
    def __Init__(self):
        self._df = None
    #get user_artists dataset    
    def get_user_artist_data(self, datafilepath: Path):
        user_artist_data = pd.read_csv(datafilepath, sep='\t')
        return user_artist_data          
    #get artists dataset
    def get_artist_data(self, datafilepath: Path):
        artists_data = pd.read_csv(datafilepath, sep='\t')
        artists_data.rename(columns={'id':'artistID', 'name':'artistName'}, inplace=True)
        artists_data.drop(columns='pictureURL', inplace=True)
        artists_data = artists_data.set_index('artistID')
        return artists_data
    #combine them both
    def get_combined_dataset(self, df1, df2):
        df = pd.merge(df1, df2, on='artistID')
        self._df = df
        return self._df

#class for preparing data for collaborative filtering    
class collaborative_data:
    
    def __init__(self):
        self._artists_df = None
        
    #function for generating csr matrix
    def get_user_artist_data_csr(self, datafilepath: Path):
        user_artist_data = pd.read_csv(datafilepath, sep='\t')
        user_artist_data.set_index(['userID', 'artistID'], inplace=True)
        coo = coo_matrix(
            (
                user_artist_data.weight.astype(float),
                (
                    user_artist_data.index.get_level_values(0),
                    user_artist_data.index.get_level_values(1),
                ),
            )
        )
        return coo.tocsr() 
    #function to get and format the artists dataset
    def load_artists(self, datafilepath: Path):
        artists_data = pd.read_csv(datafilepath, sep='\t')
        artists_data.rename(columns={'id':'artistID', 'name':'artistName'}, inplace=True)
        artists_data.drop(columns='pictureURL', inplace=True)
        artists_data = artists_data.set_index('artistID')
        self._artists_df = artists_data
    #function for getting artists name from its index
    def get_artist_name(self, artist_id: int):
        artist_name = self._artists_df.loc[artist_id, "artistName"]
        return artist_name
                      
                         
        