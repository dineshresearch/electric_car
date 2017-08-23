'''
DESCRIPTION:
    feature_engineering.py is a script that reads the raw data in, engineers
    features for the model, and generates training and testing data sets
    that are used in the construction of the machine learning models in this project.
    This script is called in several other files in this project with the intention
    of streamlining the feature engineering process.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import calendar

# enumeration of the 2,880 time increments
increments = np.arange(2880)
# assign a day number (0 to 59) for all time increments
day_nums = increments/48
# assign an integer (0 to 6) for all time increments so that the first day is Friday
day_indices = (4 + day_nums)%7
# convert the day indices into strings abbreviating the name of the days
day_names = np.array( [list(calendar.day_abbr)[n] for n in day_indices] )
# boolean vector with 1 values if the time increment corresponds to a weekend day
weekend = np.array( [1 if day in ["Sat", "Sun"] else 0 for day in day_names] )
#
df_dow = pd.DataFrame(np.vstack((day_names,weekend)).T)
df_dow[1] = df_dow[1].apply(int)
df_dow = pd.get_dummies(df_dow)

def derivative_matrix(x):
    '''
    INPUT:
        - x: M-dimensional vector (ndarray)
    OUTPUT:
        - xx: 7xM matrix containing x and numerical approximations to the first
            two derivatives using the forward, backward, and inside methods (ndarray)
    '''
    # rows = ['f', 'd1_forward', 'd1_back', 'd1_inside', 'd2_forward', 'd2_back', 'd2_inside']
    xx = np.zeros((7, len(x))) # initalize xx
    xx[0,:] = x # set first row equal to x
    xx[1, :-1] = x[1:] - x[:-1] # forward first derivative
    xx[2, 1:] = x[1:] - x[:-1] # backward first derivative
    xx[3, 1:-1] = (x[2:] - x[:-2])/2. # inside first derivative
    xx[4, :-2] = xx[1,1:-1] - xx[1,:-2] # forward second derivative
    xx[5, 2:] = xx[2,2:] - xx[2,1:-1] # backward second derivative
    xx[6, 2:-2] = (xx[3, 3:-1] - xx[3,1:-3])/2. # inside second derivative

    return xx

def transform_feature_matrix(X):
    '''
    INPUT:
        - X: NxM matrix (ndarray)
    OUTPUT:
        - XX: N*Mx7 matrix containing the dervative matrix for each row of X (ndarray)
    '''
    XX = np.zeros((X.shape[0]*X.shape[1], 7)) # initalize XX
    # loop through each row of X
    for row in xrange(X.shape[0]):
        # run derivative_matrix() on each row of X and place the resulting matrix
        # into the appropriate position in XX
        XX[row*X.shape[1]:(row+1)*X.shape[1], :] = derivative_matrix(X[row]).T
    return XX

def transform_X(X):
    '''
    INPUT:
        - X: table where the rows contain house ids and power readings (pandas DataFrame)
    OUTPUT:
        - XX: X after removing the house id column and then each row is scaled
            as standard random normal (ndarray)
        - X_transform: output of transform_feature_matrix() applied to XX (ndarray)
        - ids: list of house ids contained in X (pandas Series)
    '''
    ids = X.pop("House ID")
    XX = StandardScaler().fit_transform(X.T).T
    X_transform = transform_feature_matrix(XX)
    return (XX, X_transform, ids)

def transform_Xy(X, y):
    '''
    INPUT:
        - X: table where the rows contain house ids and power readings (pandas DataFrame)
        - y: table where the rows contain house ids and charge labels (pandas DataFrame)
    OUTPUT:
        - XX: see transform_X()
        - X_transform: see transform_X()
        - y: same as y input variable with the house id column removed (pandas DataFrame)
        - y_transform: electric vehicle charge labels reshaped into a vector (ndarray)
        - ids: see transform_X()
    '''
    (XX, X_transform, ids) = transform_X(X)
    y.pop("House ID")
    y_transform = y.values.reshape(y.shape[0]*y.shape[1])
    return (XX, X_transform, y, y_transform, ids)

def generate_output_df(y_vector, X, ids, columns):
    '''
    INPUT:
        - y_vector: vector containing the probabilities of an electric vehicle
            to be charging for a set of houses at the given time intervals (ndarray)
        - X: table where the rows contain power readings (ndarray)
        - ids: list of house ids contained in X (pandas Series)
        - columns: list of column names for the output data frame (list)
    OUTPUT:
        - df: data frame containing the house ids and probabilities that an
            electric vehicle is charging for each house at each time interval
    '''
    # reshape y_vector into a matrix
    y_matrix = y_vector.reshape(X.shape)
    # add a column containing the ids for the houses that y_matrix represents
    y_matrix = np.hstack( (np.array([ids]).T, y_matrix) )
    # create a data frame from y_matrix
    df = pd.DataFrame(y_matrix)
    # rename the columns of df
    df.columns = columns
    # set the house id column to integer type
    df["House ID"] = df["House ID"].apply(int)
    return df

# load the training power readings into a data frame
features = pd.read_csv("data/EV_train.csv")
# load the training labels into a data frame
labels = pd.read_csv("data/EV_train_labels.csv")

# extract the columns names from the features data frame
columns = features.columns
# replace nan values in features data frame with median of each row
features = features.T.fillna(features[features.columns[1:]].T.median()).T
# split the features and labels into train/test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=0.2, random_state=42)
# engineer derivative features and transform train/test feature and label matrices
# into appropriate format for building ML models
(X_train, X_train_transform, y_train, y_train_transform, train_ids) = transform_Xy(X_train, y_train)
(X_test, X_test_transform, y_test, y_test_transform, test_ids) = transform_Xy(X_test, y_test)

# load test power readings into a data frame.  this should not be confused with
# the test data above, this is the test data with no labels that is the ultimate
# goal to predict probabilities of electric vehicles charging for in this project
test_features = pd.read_csv("data/EV_test.csv")
# replace nan values with median of each row
test_features = test_features.T.fillna(test_features[test_features.columns[1:]].T.median()).T
# transform test_features into format that ML models can predict on
(X, X_transform, ids) = transform_X(test_features)
