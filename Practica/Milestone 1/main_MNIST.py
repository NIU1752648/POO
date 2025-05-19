"""
POO - Milestone 1
Random Forest Classifier
"""

# External dependencies
from sklearn.datasets import fetch_openml
from time import time

# Internal dependencies
from Logger import info
from Impurity import Impurities
from RandomForestEvaluator import RandomForestEvaluator, RandomForestClassifier
from RandomForestCExtras import RandomForestExtraTrees, RandomForestPEC

if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    print(y)
    random_forest_gini = RandomForestEvaluator(
        "Mnist",
        RandomForestPEC(num_trees = 100, min_size = 2, max_depth = 10),
        X, y
    )
    time_start = time()
    random_forest_gini.train()
    time_elapsed = time() - time_start
    info(f"Time elapsed while making trees {time_elapsed}")

    random_forest_gini.plot_accuracy()
    random_forest_gini.print_trees()
    print(random_forest_gini.feature_importance())
    random_forest_gini.plot_fi_mnist()