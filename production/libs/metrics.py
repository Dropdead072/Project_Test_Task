import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from typing import Callable, List

def apk(relevant: List[int], predicted: List[int], k: int) -> float:
    if len(predicted) > k:
        predicted = predicted[:k]
        
    score = 0
    num_hits = 0

    for i, p in enumerate(predicted):
        if p in relevant:
            num_hits += 1
            score += num_hits / (i + 1)

    return score / min(len(relevant), k)

def mapk(relevant: List[List[int]], predicted: List[List[int]], k: int = 20):
    ap_list = [apk(r, p, k) for r, p in zip(relevant, predicted)]
    return np.mean(ap_list)


def jaccard(ratings: np.array, user_vector: np.array) -> np.array:
    user_vector = user_vector.reshape(1, -1)
    and_matrix = np.logical_and(ratings, user_vector)
    or_matrix = np.logical_or(ratings, user_vector)

    distance_vector = np.sum(and_matrix, axis=1) / np.sum(or_matrix, axis=1)
    distance_vector[distance_vector == 1.] = 0
    
    return distance_vector


if __name__ == '__main__':
    rt = np.array( [[1, 0, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 1, 0], [1, 1, 1, 0, 0, 1, 0]] )
    uv = np.array( [1, 1, 1, 0, 0, 1, 0] )

    print(f'rt vectors: {rt}')
    print(f'uv vector: {uv}')
    print(f'Jaccard metric on vectors from rt to uv vector: {jaccard(rt, uv)}')

