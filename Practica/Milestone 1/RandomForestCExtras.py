import numpy as np
import multiprocessing

from Dataset import Dataset
from RandomForestClassifier import RandomForestClassifier

def _worker(forest, dataset, tree_index):
    forest._make_tree(dataset)

class RandomForestParallelism(RandomForestClassifier):
    def _make_decision_trees(self, dataset: Dataset):
        """Hola que tal"""
        with multiprocessing.Manager() as manager:
            shared_trees = manager.list()

            original_trees = self.decision_trees
            self.decision_trees = shared_trees

            with multiprocessing.Pool() as pool:
                pool.map(_worker, range(self.num_trees))

            self.decision_trees = original_trees
            self.decision_trees.extend(list(shared_trees))


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

class RandomForestPEC(RandomForestClassifier):
    def _make_decision_trees(self, dataset: Dataset):
        with multiprocessing.Manager() as manager:
            shared_trees = manager.list()
            original_trees = self.decision_trees
            self.decision_trees = shared_trees

            with multiprocessing.Pool() as pool:
                pool.starmap(_worker, [(self, dataset, i) for i in range(self.num_trees)])

            self.decision_trees = original_trees
            self.decision_trees.extend(list(shared_trees))

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