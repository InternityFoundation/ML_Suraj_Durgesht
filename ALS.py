# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 23:47:26 2019

@author: Suraj
"""


import pandas as pd
import sys
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random

from sklearn.preprocessing import MinMaxScaler

import implicit


u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users =  pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')


print(users.shape)
print(users.head())

print(ratings.shape)
print(ratings.head())


print(items.shape)
print(items.head())

dataset = pd.merge(pd.merge(items, ratings),users)
print(dataset.head())

sparse_item_user = sparse.csr_matrix((dataset['rating'].astype(float),(dataset['movie_id'], dataset['user_id'])))
sparse_user_item = sparse.csr_matrix((dataset['rating'].astype(float),(dataset['user_id'], dataset['movie_id'])))

model = implicit.als.AlternatingLeastSquares(factors=20,regularization=0.1,iterations=20)
alpha_val = 15
data_conf = (sparse_item_user * alpha_val).astype('double')
model.fit(data_conf)
item_id = 22
n_similar = 5
similar = model.similar_items(item_id,n_similar)
for item in similar:
    idx,score = item
    print(dataset.movie_title.loc[dataset.movie_id == idx].iloc[0])

user_id = 936
recommended = model.recommend(user_id,sparse_user_item)
movies = []
scores = []

for item in recommended:
    idx, score = item
    movies.append(dataset.movie_title.loc[dataset.movie_id == idx].iloc[0])
    scores.append(score)
    
recommendations = pd.DataFrame({'movies': movies, 'scores':scores})
print(recommendations)
