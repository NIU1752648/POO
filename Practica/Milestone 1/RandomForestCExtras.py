import numpy as np
from multiprocessing import Pool

from Dataset import Dataset
from RandomForestClassifier import RandomForestClassifier

class RandomForestParallelism(RandomForestClassifier):
    def _make_decision_trees(self, dataset: Dataset):
        super()._make_tree(dataset)
        pass

class RandomForestExtraTrees(RandomForestClassifier):
    def _best_split(self, dataset: Dataset, idx_features: np.array):
        best_feature_index, best_threshold, minimum_cost, best_split = \
            np.inf, np.inf, np.inf, None
        for idx in idx_features:
            random_value = np.random.choice(dataset.X[:, idx])
            left_dataset, right_dataset = dataset.split(idx, random_value)
            cost = self._cart_cost(left_dataset, right_dataset, dataset)
            if cost < minimum_cost:
                best_feature_index, best_threshold, minimum_cost, best_split = \
                    idx, random_value, cost, [left_dataset, right_dataset]
        return best_feature_index, best_threshold, minimum_cost, best_split