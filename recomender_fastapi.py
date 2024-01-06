from fastapi import FastAPI
import os
import requests
from pydantic import BaseModel
import implicit
from model import Popularity, Implicit_recommender
from data import popularity_data, collaborative_data


#'RecIn' Class for Popularity recommender model
class RecIn(BaseModel):
    user_id : int
# 'RecOut' class for popularity recommender model        
class RecOut(RecIn):
    success: bool
    message: str
    user_recommendations: dict

# 'Rec2In' Class for implicit recommender model        
class Rec2In(BaseModel):
    user_id: int
    n_recommendations: int
# 'Rec2Out' Class for implicit recommender mode        
class Rec2Out(Rec2In):
    success: bool
    message: str
    recommendations: dict

#instantiate fastapi
app = FastAPI()

#create a "/hello" path with 200 status code
@app.get("/hello", status_code=200)
def hello():
    return{"message": "holla bro"}

#create a '/rec' path with 200 status code for popularity recommenders
@app.post("/rec", status_code=200, response_model=RecOut)
def popularity_recommender(request:RecIn):
    #create response dictionary from requests
    response = request.dict()
    model = Popularity()
    #create model
    model.create()
    #save model
    filename = model.dump()
    #add success key
    response['success'] = True
    #add message key
    response['message'] = f"popularity model stored at '{filename}'."
    #load stored model
    model.load()
    #show user recommendations
    recommendations = model.recommend(request.user_id)
    response['user_recommendations'] = recommendations
    return response
    
#create a '/rec2' path with 200 status code for popularity recommenders
@app.post('/rec2', status_code=200, response_model=Rec2Out)
def als(request:Rec2In):
    #response dictionary for requests
    response = request.dict()
    #instantiate data for collaborative filtering
    data = collaborative_data()
    #load user_artists matrix
    user_artists = data.get_user_artist_data_csr("./dataset/user_artists.dat")
    #load artist retriever data
    artists_data = data.load_artists("./dataset/artists.dat")
    #instantiate ALS using implicit
    ALS = implicit.als.AlternatingLeastSquares(
    factors=50, iterations=10, regularization=0.01
    )
    #instantiate recommender
    recommender = Implicit_recommender(artist_retriever=data, implicit_model=ALS)
    #fit user_artists data
    recommender.fit(user_artists)
    #save the recommender model
    filename = recommender.dump()
    #add success key
    response['success'] = True
    #add message key
    response['message'] = f"ALS model prepared and stored at '{filename}'."
    #load recommender model
    recommender.load()
    #perform user recommendations
    artists, scores = recommender.recommend(request.user_id, user_artists, request.n_recommendations)
    recommendations_dict = {}
    #iterate through
    for artists, score in zip(artists, scores):
        recommendations_dict[f'{artists}'] = '{:0.2f}'.format(score)
    response['recommendations'] = recommendations_dict
        
    return response
    
    
    
    
    
    
    
