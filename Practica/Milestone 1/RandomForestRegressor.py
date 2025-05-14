import numpy as np

from RandomForestClassifier import RandomForestClassifier
from Impurity import Impurities
from Dataset import Dataset
from Tree import Tree

class RandomForestRegressor(RandomForestClassifier):
    def __init__(self, num_trees = 100, ratio_samples = 0.8, max_depth = 10, impurity = Impurities.SumSquareError(), min_size = 1):
        super().__init__(num_trees, ratio_samples, max_depth, impurity, min_size)
    
    @staticmethod
    def _make_leaf(dataset:Dataset):
        return Tree.Leaf(dataset.mean_value())

    @staticmethod
    def _combine_predictions(prediccions: list):
        return np.mean(prediccions)