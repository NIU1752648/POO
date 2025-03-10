from Parent import Parent
from Leaf import Leaf
from Dataset import Dataset
import numpy as np

class RandomForest:
    def __init__(self):
        self.num_trees = 100
        self.ratio_samples = 1.0
        self.max_depth = 10
        self.min_size = 1
        self.num_features = 0
        self.decision_trees = []

    def fit (self, X: np.array, y: np.array):
        # X is a matrix of size (num_samples, num_features)
        # y is a vector of size (num_samples, 1)
        # a pair (X,y) is a dataset, with its own responasibilities
        dataset=Dataset(X,y)
        self._make_decision_trees(dataset)

    def _make_decision_trees(self, dataset: Dataset):
        self.decision_trees = []
        for i in range(self.num_trees):
            # sample a subset of the dataset with replacement using
            # np.random.choice() to get the indices of rows in X and y
            subset=dataset.random_sampling(self.ratio_samples)
            tree=self._make_node(subset,1) # the root of the decision tree
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
    
    def _make_leaf(self, dataset: Dataset):
        # label = most frequent class in dataset
        return Leaf(dataset.most_frequent_label())
    
    def _make_parent_or_leaf(self, dataset: Dataset, depth: int):
        # select a random subset of features, to make trees more diverse
        idx_features = np.random.choice(dataset.num_features,
                                        self.num_features,replace=False)
        best_feature_index, best_threshold, minimum_cost, best_split = \
            self._best_split(dataset,idx_features)
        
        left_dataset, right_dataset = self._best_split(dataset, idx_features)
        assert left_dataset.num_samples > 0 or right_dataset.num_samples > 0
        if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
            # this is an special case: dataset has samples of at least two
            # classes but the best split is moving all samples to the left or right
            # datset and none to the other, so we make leaf instead of a parent
            return self._make_leaf(dataset)
        else:
            node = Parent(best_feature_index,best_threshold)
            node.left_child = self._make_node(left_dataset,depth+1)
            node.right_child = self._make_node(right_dataset,depth+1)
            return node
        
    def _best_split(self, dataset: Dataset, idx_features: int):
        # find the best pair (feature, threshold) by exploring all possible pairs
        best_feature_index, best_threshold, minimum_cost, best_split = \
            np.Inf, np.Inf, np.Inf, None
        for idx in idx_features:
            values = np.unique(dataset.X[:,idx])
            for val in values:
                left_dataset, right_dataset = dataset.split(idx,val)
                cost = self._CART_cost(left_dataset,right_dataset)
                if cost < minimum_cost:
                    best_feature_index, best_threshold, minimum_cost, best_split = \
                        idx, val, cost, [left_dataset,right_dataset]
        return best_feature_index, best_threshold, minimum_cost, best_split
    
    @staticmethod
    def _CART_cost(left_dataset: Dataset, right_dataset: Dataset, dataset: Dataset):
        # the J(k,v) equation, using Gini
        cost = (left_dataset.num_samples/dataset.num_samples)*RandomForest._gini(left_dataset)\
            + (right_dataset.num_samples/dataset.num_samples)*RandomForest._gini(right_dataset)
        return cost 
    
    @staticmethod
    def _gini(dataset: Dataset):
        # the Gini impurity equation
        for label in np.unique(dataset.y):
            count = 0
            for elem in dataset.y:
                if elem == label:
                    count += 1
            gini += (count/dataset.num_samples)**2
        gini = 1 - gini
        return gini