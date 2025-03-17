import numpy as np
import matplotlib.pyplot as plt

from Tree import Tree
from RandomForestClassifier import RandomForestClassifier

class RandomForestEvaluator:
    def __init__(self, random_forest: RandomForestClassifier, X: np.array,
                 y: np.array, ratio_train = 0.7, ratio_test = 0.3):
        self._random_forest = random_forest

        self.ratio_train, self.ratio_test = ratio_train, ratio_test
        self.num_samples, self.num_features = X.shape

        idx = np.random.permutation(range(self.num_samples))

        self.num_samples_train = int(self.num_samples * ratio_train)
        self.num_samples_test = int(self.num_samples * ratio_test)
        self.idx_train = idx[:self.num_samples_train]
        self.idx_test = idx[self.num_samples_train: self.num_samples_train + self.num_samples_test]
        self.X_train, self.y_train = X[self.idx_train], y[self.idx_train]
        self.X_test, self.y_test = X[self.idx_test], y[self.idx_test]

    @property
    def random_forest(self):
        return self._random_forest

    def train(self):
        self.random_forest.fit(self.X_train, self.y_train)

    @property
    def evaluate(self):
        accuracy = list()
        for tree_idx, tree in enumerate(self.random_forest.decision_trees):
            accuracy.append(0)
            for row, sample in enumerate(self.X_test):
                parent = tree
                while isinstance(parent, Tree.Parent):
                    if parent.predict(sample):
                        parent = parent.left_child
                    else:
                        parent = parent.right_child
                if parent.predict(sample) == self.y_test[row]:
                    accuracy[tree_idx] += 1
            accuracy[tree_idx] /= self.num_samples_test
        return accuracy

    def plot_accuracy(self):
        accuracy = self.evaluate
        plt.bar(range(1, len(accuracy) + 1, ), accuracy, color='blue')

        plt.title(f'Random Forest ({self.random_forest.impurity})')
        plt.xlabel('Tree')
        plt.ylabel('Accuracy')

        plt.show()