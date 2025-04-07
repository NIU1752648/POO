
# External dependencies
import pandas as pd
import numpy as np

# Internal dependencies
from Impurity import Impurities
from RandomForestEvaluator import RandomForestEvaluator, RandomForestClassifier

def load_sonar():
    try:
        df = pd.read_csv('sonar.all-data.csv' ,header=None)
    except FileNotFoundError:
        df = pd.read_csv('Practica/Milestone 1/sonar.all-data.csv', header=None)
    X = df[df.columns[:-1]].to_numpy()
    y = df[df.columns[-1]].to_numpy(dtype=str)
    y = (y=='M').astype(int) # M = mine, R = rock
    return X, y

if __name__ == "__main__":
    X, y = load_sonar()

    random_forest_gini = RandomForestEvaluator(
        "Sonar",
        RandomForestClassifier(),
        X, y
    )

    random_forest_entropy = RandomForestEvaluator(
        "Sonar",
        RandomForestClassifier(impurity=Impurities.Entropy()),
        X, y
    )

    random_forest_gini.train()
    random_forest_entropy.train()

    random_forest_gini.plot_accuracy()
    random_forest_entropy.plot_accuracy()