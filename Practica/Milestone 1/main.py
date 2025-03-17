"""
POO - Milestone 1
Random Forest Classifier
"""

# External dependencies
import numpy as np
import sklearn.datasets

# Internal dependencies
from Impurity import Impurities
from Tree import Tree
from RandomForestEvaluator import RandomForestEvaluator, RandomForestClassifier

if __name__ == "__main__":
    iris = sklearn.datasets.load_iris()
    X, y = iris.data, iris.target

    random_forest_gini = RandomForestEvaluator(
        RandomForestClassifier(),
        X, y
    )

    random_forest_entropy = RandomForestEvaluator(
        RandomForestClassifier(impurity = Impurities.Entropy()),
        X, y
    )

    random_forest_entropy.train()

    random_forest_entropy.plot_accuracy()
