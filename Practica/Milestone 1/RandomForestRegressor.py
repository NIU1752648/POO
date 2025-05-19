import numpy as np

from RandomForestClassifier import RandomForestClassifier
from Impurity import Impurities
from Dataset import Dataset
from Tree import Tree

class RandomForestRegressor(RandomForestClassifier):
    def __init__(self, num_trees = 100, ratio_samples = 0.8, max_depth = 10, impurity = Impurities.SumSquareError(), min_size = 1):
        # Changing some parameters to make it a regressor and more acurate
        super().__init__(num_trees, ratio_samples, max_depth, impurity, min_size)
    
    @staticmethod
    def _make_leaf(dataset:Dataset):
        # (change the function of RandomForestClassifier)
        # Return a leaf node with the mean value of the target variable
        return Tree.Leaf(dataset.mean_value())

    @staticmethod
    def _combine_predictions(prediccions: list):
        # Combine the predictions of the trees by taking the mean
        return np.mean(prediccions)