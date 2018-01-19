import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engineering import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

if __name__ == '__main__':
    lr = LogisticRegression().fit(X_train_transform, y_train_transform)
    # logistic_regression = LogisticRegression(class_weight="balanced").fit(X_train_transform, y_train_transform)

    print "Train Accuracy: {0}".format(lr.score(X_train_transform, y_train_transform))
    print "Test Accuracy: {0}".format(lr.score(X_test_transform, y_test_transform))

    from feature_engineering2 import *
    lr2 = LogisticRegression().fit(X_train_transform, y_train_transform)
    print "Train Accuracy: {0}".format(lr2.score(X_train_transform, y_train_transform))
    print "Test Accuracy: {0}".format(lr2.score(X_test_transform, y_test_transform))
