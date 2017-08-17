import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
import calendar

def derivative_matrix(x):
    # rows = ['f', 'd1_forward', 'd1_back', 'd1_inside', 'd2_forward', 'd2_back', 'd2_inside']
    xx = np.zeros((7, len(x)))
    xx[0,:] = x
    xx[1, :-1] = x[1:] - x[:-1]
    xx[2, 1:] = x[1:] - x[:-1]
    xx[3, 1:-1] = (x[2:] - x[:-2])/2.
    xx[4, :-2] = xx[1,1:-1] - xx[1,:-2]
    xx[5, 2:] = xx[2,2:] - xx[2,1:-1]
    xx[6, 2:-2] = (xx[3, 3:-1] - xx[3,1:-3])/2.

    return xx

def expand_derivative_matrix(x):
    # rows = ['f', 'd1_forward', 'd1_back', 'd1_inside', 'd2_forward', 'd2_back', 'd2_inside',
    #         'weekend', 'day_Fri', 'day_Mon', 'day_Sat', 'day_Sun', 'day_Thu', 'day_Tue', 'day_Wed']
    indices = np.arange(2880)
    day_nums = indices/48
    day_indices = (4 + day_nums)%7
    day_names = np.array( [list(calendar.day_abbr)[n] for n in day_indices] )
    weekend = np.array( [1 if day in ["Sat", "Sun"] else 0 for day in day_names] )
    df = pd.DataFrame(np.vstack((day_names,weekend)).T)
    df[1] = df[1].apply(int)
    df = pd.get_dummies(df)
    return np.vstack((derivative_matrix(x), df.values.T))


def transform_feature_matrix(X):
    XX = np.zeros((X.shape[0]*X.shape[1], 15))
    for row in xrange(X.shape[0]):
        XX[row*X.shape[1]:(row+1)*X.shape[1], :] = expand_derivative_matrix(X[row]).T
    return XX

if __name__ == '__main__':
    features = pd.read_csv("data/EV_train.csv")
    labels = pd.read_csv("data/EV_train_labels.csv")

    features.pop("House ID")
    labels.pop("House ID")

    features = features.T.fillna(features.T.median()).T

    X_train, X_test, y_train, y_test = train_test_split(features.values, labels.values,
                                                        test_size=0.2, random_state=42)

    train_car_indices = np.where(np.sum(y_train, axis=1)>0)[0]
    X_train = X_train[train_car_indices]
    y_train = y_train[train_car_indices]

    # neigh = NearestNeighbors(n_neighbors = 5)
    # neigh.fit(X_train)
    # nearest = neigh.kneighbors(X_test[1].reshape(1,-1))

    # models = []
    # for n in xrange(X_train.shape[0]):
    #     models.append(LogisticRegression().fit(X_train[n].reshape(-1,1), y_train[n]))
        # models.append(LogisticRegression().fit(derivative_matrix(X_train[n]).T, y_train[n]))

    indices = np.arange(2880)
    day_nums = indices/48
    day_indices = (4 + day_nums)%7
    day_names = np.array( [list(calendar.day_abbr)[n] for n in day_indices] )
    weekend = np.array( [1 if day in ["Sat", "Sun"] else 0 for day in day_names] )
