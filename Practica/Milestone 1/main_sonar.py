
# External dependencies
import pandas as pd
import numpy as np

# Internal dependencies
from Impurity import Impurities
from RandomForestEvaluator import RandomForestEvaluator, RandomForestClassifier

def load_sonar():
    df = pd.read_csv('sonar.all-data.csv' ,header=None)
    X = df[df.columns[:-1]].to_numpy()
    y = df[df.columns[-1]].to_numpy(dtype=str)
    y = (y=='M').astype(int) # M = mine, R = rock
    return X, y

if __name__ == "__main__":
    X, y = load_sonar()

    random_forest_gini = RandomForestEvaluator(
        "Sonar",
        RandomForestClassifier(max_depth = 15),
        X, y
    )

    random_forest_entropy = RandomForestEvaluator(
        "Sonar",
        RandomForestClassifier(max_depth = 15, impurity=Impurities.Entropy()),
        X, y
    )

    random_forest_gini.train()
    random_forest_entropy.train()

    random_forest_gini.plot_accuracy()
    random_forest_entropy.plot_accuracy()