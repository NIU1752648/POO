from sklearn.datasets import fetch_openml

from RandomForestEvaluator import RandomForestEvaluator
from RandomForestCExtras import RandomForestParallelism

if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    random_forest_gini = RandomForestEvaluator(
        "Mnist",
        RandomForestParallelism(),
        X, y
    )

    random_forest_gini.train()

    random_forest_gini.plot_accuracy()