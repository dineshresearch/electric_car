import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engineering import *
from sklearn.ensemble import RandomForestClassifier
from pickle_io import *

if __name__ == '__main__':
    # random_forest = RandomForestClassifier(n_estimators = 5).fit(X_train_transform, y_train_transform)
    #
    # print "Train Accuracy: {0}".format(random_forest.score(X_train_transform, y_train_transform))
    # print "Test Accuracy: {0}".format(random_forest.score(X_test_transform, y_test_transform))

    rf = pickle_load("models/random_forest_30.pkl")
    df = generate_output_df(rf.predict_proba(X_transform)[:,1], X, ids, columns)
    df.to_csv("data/random_forest_submission.csv", index=False)
