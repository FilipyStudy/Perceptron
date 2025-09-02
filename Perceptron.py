import pandas as pd
import numpy as np
from pandas import DataFrame

rng = np.random.default_rng()

#Return -1, 0 or 1 using the sign() mathematical function
def binary_classification(df):
    #Variables definition
    len_X = df.shape[1]
    weights = np.ndarray((df.shape[1], 1), dtype=np.float64)
    y_predicted = np.ndarray((df.shape[1], 1), dtype=np.float64)
    np.append(weights, np.random.uniform(-1.0, 1.0, (1, len_X)))
    iterator = df.iterrows()

    #Iterate over each row
    for index, row in iterator:
        y_predicted = np.multiply(row[index], weights)
        if np.sum(y_predicted) >= 0:
            weights = np.subtract(weights, row)
        elif np.sum(y_predicted) < 0:
            weights = np.sum(weights, row)
    return weights


class SingleLayer:
    def __init__(self, single_data):
        self.single_data = single_data


class Multilayer:
    def __init__(self, multi_data):
            self.multi_data = multi_data