"""
POO - Milestone 1
Random Forest Classifier
"""

# External dependencies
import sklearn.datasets

# Internal dependencies
from Impurity import Impurities
from RandomForestEvaluator import RandomForestEvaluator, RandomForestClassifier

if __name__ == "__main__":
    iris = sklearn.datasets.load_iris()
    X, y = iris.data, iris.target

    random_forest_gini = RandomForestEvaluator(
        "Iris",
        RandomForestClassifier(),
        X, y
    )

    random_forest_entropy = RandomForestEvaluator(
        "Iris",
        RandomForestClassifier(impurity = Impurities.Entropy()),
        X, y
    )

    random_forest_gini.train()

    random_forest_gini.plot_fi()
