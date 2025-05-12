from RandomForestClassifier import RandomForestClassifier
from Impurity import Impurities

class RandomForestRegressor(RandomForestClassifier):
    def __init__(self, num_trees = 100, ratio_samples = 0.8, max_depth = 10, impurity = Impurities.SumSquareError(), min_size = 1):
        super().__init__(num_trees, ratio_samples, max_depth, impurity, min_size)

