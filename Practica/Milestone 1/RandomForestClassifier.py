import numpy as np
from tqdm import tqdm

import Logger
from Tree import Tree
from Dataset import Dataset
from Impurity import Impurities

@Logger.log_instance
class RandomForestClassifier:
    def __init__(self, num_trees = 100, ratio_samples = 0.8, max_depth = 10, impurity = Impurities.GiniIndex(), min_size = 1):
        self.num_trees = num_trees # Number of trees in the forest
        self.ratio_samples = ratio_samples # Percentage of uses of the training set for training each tree (1.0 = 100%)
        self.max_depth = max_depth # Maximum depth of the decision trees
        self.min_size = min_size # Minimum number of samples in a node to be considered for splitting
        self.num_features = 1 # Number of features to consider when looking for the best split
        self.decision_trees = []
        self.impurity = impurity

    def fit(self, X: np.array, y: np.array):
        # X is a matrix of size (num_samples, num_features)
        # y is a vector of size (num_samples, 1)
        # a pair (X,y) is a dataset, with its own responsibilities
        dataset = Dataset(X, y)
        self._make_decision_trees(dataset)

    def _make_decision_trees(self, dataset: Dataset):
        for i in tqdm(range(self.num_trees), desc=f"Planting trees ({str(self.impurity)})", ascii=False, ncols=100):
            self._make_tree(dataset)

    def _make_tree(self, dataset):
        # sample a subset of the dataset with replacement using
        # np.random.choice() to get the indices of rows in X and y
        subset = dataset.random_sampling(self.ratio_samples)
        tree = self._make_node(subset, 1)  # the root of the decision tree
        self.decision_trees.append(tree)

    def _make_node(self, dataset: Dataset, depth: int):
        if depth == self.max_depth\
            or dataset.num_samples <= self.min_size\
            or len(np.unique(dataset.y)) == 1:
            # last condition is true if all samples belong to the same class
            node = self._make_leaf(dataset)
        else:
            node = self._make_parent_or_leaf(dataset,depth)
        return node

    @staticmethod
    def _make_leaf(dataset: Dataset):
        # label = most frequent class in dataset
        return Tree.Leaf(dataset.most_frequent_label())
    
    def _make_parent_or_leaf(self, dataset: Dataset, depth: int):
        # select a random subset of features, to make trees more diverse
        idx_features = np.random.choice(range(dataset.num_features), self.num_features, replace=False)
        best_feature_index, best_threshold, minimum_cost, best_split = \
            self._best_split(dataset,idx_features)
        left_dataset, right_dataset = best_split

        assert left_dataset.num_samples > 0 or right_dataset.num_samples > 0
        if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
            # this is a special case: dataset has samples of at least two
            # classes but the best split is moving all samples to the left or right
            # dataset and none to the other, so we make leaf instead of a parent
            return self._make_leaf(dataset)
        else:
            node = Tree.Parent(best_feature_index,best_threshold)
            node.left_child = self._make_node(left_dataset,depth+1)
            node.right_child = self._make_node(right_dataset,depth+1)
            return node
        
    def _best_split(self, dataset: Dataset, idx_features: np.array):
        # find the best pair (feature, threshold) by exploring all possible pairs
        best_feature_index, best_threshold, minimum_cost, best_split = \
            np.inf, np.inf, np.inf, None
        for idx in idx_features:
            values = np.unique(dataset.X[:,idx])
            for val in values:
                left_dataset, right_dataset = dataset.split(idx,val)
                cost = self._cart_cost(left_dataset,right_dataset, dataset)
                if cost < minimum_cost:
                    best_feature_index, best_threshold, minimum_cost, best_split = \
                        idx, val, cost, [left_dataset,right_dataset]
        return best_feature_index, best_threshold, minimum_cost, best_split
    
    def _cart_cost(self, left_dataset: Dataset, right_dataset: Dataset, dataset: Dataset):
        cost = 0
        # the J(k,v) equation
        cost = (left_dataset.num_samples/dataset.num_samples)*self.impurity.impurity(left_dataset)\
            + (right_dataset.num_samples/dataset.num_samples)*self.impurity.impurity(right_dataset)
        return cost

    @staticmethod
    def _combine_predictions(prediccions: list):
        return np.argmax(np.bincount(prediccions))

    def predict(self, X_test: np.array):
        y_pred = list()
        for x in X_test:
            prediccions = list()
            for tree in self.decision_trees:
                parent = tree
                while isinstance(parent, Tree.Parent):
                    if parent.predict(x):
                        parent = parent.left_child
                    else:
                        parent = parent.right_child
                prediccions.append(parent.predict(x))
            y_pred.append(self._combine_predictions(prediccions))
        return np.array(y_pred)