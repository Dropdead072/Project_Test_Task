import pandas as pd
import numpy as np
import mlflow
from libs.metrics import *
from libs.processing import *
from libs.model import User2User


def data_loading(path: str, user_file: str, track_file: str):
    user_data = pd.read_csv(f"{path}\{user_file}")
    track_data = pd.read_csv(f"{path}\{track_file}")
    return user_data, track_data


def preprocess_data(user_data, track_data):
    train_ratings, test_ratings = train_test_split(user_data)
    
    redundant_rows = np.where(~np.isin(track_data['id'], train_ratings['trackId'].unique()))[0]
    track_data.drop(redundant_rows, inplace=True)
    track_data = track_data.reset_index(drop=True)

    uencoder, iencoder = ids_encoder(train_ratings)
    train_ratings['trackId'] = iencoder.transform(train_ratings['trackId'].tolist())
    test_ratings['trackId'] = iencoder.transform(test_ratings['trackId'].tolist())
    track_data['id'] = iencoder.transform(track_data['id'].tolist())

    test_relevant = []
    test_users = []
    for user_id, user_data in test_ratings.groupby('userId'):
        test_relevant += [user_data['trackId'].tolist()]
        test_users.append(user_id)

    return train_ratings, test_ratings, test_relevant, test_users


def train_test_model(train_data, test_relevant, k=10):
    mlflow.start_run()

    model = User2User(train_data)

    pred_recs = model.get_test_recommendations(k)
    filtered_pred_recs = [pred_recs[i] for i in range(len(pred_recs)) if i in test_users]

    mlflow.log_metric("mapk", mapk(test_relevant, filtered_pred_recs, k))
    
    mlflow.end_run()



if __name__ == '__main__':
    path = input('Input path to the dir with files \n')
    user_info, track_info = list(input('Input names of user info and track info files (in name.csv format, space separated) \n').split(' '))

    user_data, track_data = data_loading(path, user_info, track_info)
    train_ratings, test_ratings, test_relevant, test_users = preprocess_data(user_data, track_data)

    num_recs = int(input('Input the number of track recomentations \n'))
    train_test_model(train_ratings, test_relevant, num_recs)
