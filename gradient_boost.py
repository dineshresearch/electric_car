import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engineering import *
from sklearn.ensemble import GradientBoostingClassifier
from pickle_io import *

if __name__ == '__main__':
    # gradient_boost = GradientBoostingClassifier().fit(X_train_transform, y_train_transform)
    #
    # print "Train Accuracy: {0}".format(gradient_boost.score(X_train_transform, y_train_transform))
    # print "Test Accuracy: {0}".format(gradient_boost.score(X_test_transform, y_test_transform))

    gb = pickle_load("models/gradient_boost.pkl")
    df = generate_output_df(gb.predict_proba(X_transform)[:,1], X, ids, columns)
    df.to_csv("data/gradient_boost_submission.csv", index=False)
