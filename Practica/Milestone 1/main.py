"""
POO - Milestone 1
Random Forest Classifier
"""

import numpy as np
import sklearn.datasets
from RandomForestClassifier import RandomForestClassifier
from Impurity import Impurities

if __name__ == "__main__":
    iris = sklearn.datasets.load_iris()
    X, y = iris.data, iris.target
    print(X.shape)


    forest = RandomForestClassifier()
    forest.max_depth = 10


    forest.fit(X, y)