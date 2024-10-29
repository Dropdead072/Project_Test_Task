import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from typing import Callable, List

from libs.metrics import *


class User2User:
    def __init__(self, ratings: pd.DataFrame, alpha=0.02):
        self.ratings = ratings
        self.n_users = len(np.unique(self.ratings['userId']))
        self.n_items = len(np.unique(self.ratings['trackId']))

        self.R = np.zeros((self.n_users, self.n_items))
        self.R[self.ratings['userId'], self.ratings['trackId']] = 1.

        self.similarity_func = jaccard
        self.alpha = alpha

    def remove_train_items(self, preds: List[List[int]], k: int):
        """
        param preds: [n_users, n_items] - recommended items for each user
        param k: int
        return: np.array [n_users, k] - recommended items without training examples
        """
        new_preds = np.zeros((len(preds), k), dtype=int)
        for user_id, user_data in self.ratings.groupby('userId'):
            user_preds = preds[user_id]
            new_preds[user_id] = user_preds[~np.in1d(user_preds, user_data['trackId'])][:k]

        return new_preds

    def get_test_recommendations(self, k: int):
        test_preds = []
        
        # your code here: (￣▽￣)/♫•*¨*•.¸¸♪
        # apply recommend along every user
        # remove train (listened items) items

        recommendation_matrix = np.zeros((self.n_users, self.n_items), dtype=int)
        for uid in range(self.n_users):
            user_recommendations = self.recommend(uid)
            recommendation_matrix[uid] = np.squeeze(user_recommendations)
        test_preds[:self.n_users] = self.remove_train_items(recommendation_matrix, k).tolist()
        
        return test_preds
    
    def similarity(self, user_vector: np.array):
        """
        user_vector: [n_items]
        """
        distance = self.similarity_func(self.R, user_vector)
        similar_objects = np.argwhere(distance >= self.alpha) # similar users in our case
        return similar_objects

    def recommend(self, uid: int):
        similar_users_index = np.squeeze(self.similarity(self.R[uid]))
        similar_users_matrix = self.R[similar_users_index]
        similar_users_distance = self.similarity_func(similar_users_matrix, self.R[uid])
        
        if similar_users_matrix.ndim == 1:
            recommended_tracks = (similar_users_distance * similar_users_matrix) / (np.abs(similar_users_distance).sum() + 1e-4)
        else:
            weighted_ratings = np.dot(similar_users_distance.T, similar_users_matrix)
            sum_similarities = np.abs(similar_users_distance).sum() + 1e-4
            recommended_tracks = weighted_ratings / sum_similarities
    
        recommended_tracks = np.argsort(-recommended_tracks)
        
        return recommended_tracks