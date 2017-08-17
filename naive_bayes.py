import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engineering import *
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
    naive_bayes = GaussianNB().fit(X_train_transform, y_train_transform)

    print "Train Accuracy: {0}".format(naive_bayes.score(X_train_transform, y_train_transform))
    print "Test Accuracy: {0}".format(naive_bayes.score(X_test_transform, y_test_transform))
