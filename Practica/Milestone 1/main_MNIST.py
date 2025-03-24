"""
POO - Milestone 1
Random Forest Classifier
"""

# External dependencies
from sklearn.datasets import fetch_openml

# Internal dependencies
from Impurity import Impurities
from RandomForestEvaluator import RandomForestEvaluator, RandomForestClassifier

if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target

    random_forest_gini = RandomForestEvaluator(
        "Mnist",
        RandomForestClassifier(),
        X, y
    )

    random_forest_entropy = RandomForestEvaluator(
        "Mnist",
        RandomForestClassifier(impurity = Impurities.Entropy()),
        X, y
    )

    random_forest_gini.train()
    random_forest_entropy.train()

    random_forest_gini.plot_accuracy()
    random_forest_entropy.plot_accuracy()