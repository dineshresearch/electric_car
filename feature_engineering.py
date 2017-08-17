import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import calendar

increments = np.arange(2880)
day_nums = increments/48
day_indices = (4 + day_nums)%7
day_names = np.array( [list(calendar.day_abbr)[n] for n in day_indices] )
weekend = np.array( [1 if day in ["Sat", "Sun"] else 0 for day in day_names] )
df_dow = pd.DataFrame(np.vstack((day_names,weekend)).T)
df_dow[1] = df_dow[1].apply(int)
df_dow = pd.get_dummies(df_dow)

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
    return np.vstack((derivative_matrix(x), df_dow.values.T))

def transform_feature_matrix(X):
    XX = np.zeros((X.shape[0]*X.shape[1], 7))
    for row in xrange(X.shape[0]):
        XX[row*X.shape[1]:(row+1)*X.shape[1], :] = derivative_matrix(X[row]).T
    return XX

def transform_feature_matrix2(X):
    XX = np.zeros((X.shape[0]*X.shape[1], 15))
    for row in xrange(X.shape[0]):
        XX[row*X.shape[1]:(row+1)*X.shape[1], :] = expand_derivative_matrix(X[row]).T
    return XX

def transform_X(X):
    ids = X.pop("House ID")
    X = StandardScaler().fit_transform(X.T).T
    X_transform = transform_feature_matrix(X)
    # X_transform = transform_feature_matrix2(X)
    return (X, X_transform, ids)

def transform_Xy(X, y):
    (X, X_transform, ids) = transform_X(X)
    y.pop("House ID")
    y_transform = y.values.reshape(y.shape[0]*y.shape[1])
    return (X, X_transform, y, y_transform, ids)

def generate_output_df(y_vector, X, ids, columns):
    y_matrix = y_vector.reshape(X.shape)
    y_matrix = np.hstack( (np.array([ids]).T, y_matrix) )
    df = pd.DataFrame(y_matrix)
    df.columns = columns
    df["House ID"] = df["House ID"].apply(int)
    return df

features = pd.read_csv("data/EV_train.csv")
labels = pd.read_csv("data/EV_train_labels.csv")

columns = features.columns
features = features.T.fillna(features[features.columns[1:]].T.median()).T
X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=0.2, random_state=42)
(X_train, X_train_transform, y_train, y_train_transform, train_ids) = transform_Xy(X_train, y_train)
(X_test, X_test_transform, y_test, y_test_transform, test_ids) = transform_Xy(X_test, y_test)


test_features = pd.read_csv("data/EV_test.csv")
test_features = test_features.T.fillna(test_features[test_features.columns[1:]].T.median()).T
(X, X_transform, ids) = transform_X(test_features)
