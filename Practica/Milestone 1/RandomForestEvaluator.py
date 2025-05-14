import os
import numpy as np
import matplotlib.pyplot as plt

import Logger
from Tree import Tree
from RandomForestClassifier import RandomForestClassifier
from RandomForestRegressor import RandomForestRegressor

class RandomForestEvaluator:
    def __init__(self, database_name: str, random_forest: RandomForestClassifier, X: np.array,
                 y: np.array, ratio_train = 0.7, ratio_test = 0.3):
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            Logger.warning("Input data contains NaN values. This may cause problems.")
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            Logger.warning("Input data contains infinite values. This may cause problems.")

        if y.shape[0] > 5000:
            Logger.warning(f"Initializing random forest with big dataset - size: {X.shape}")

        self._random_forest = random_forest
        self._database_name = database_name

        self.ratio_train, self.ratio_test = ratio_train, ratio_test
        self.num_samples, self.num_features = X.shape

        idx = np.random.permutation(range(self.num_samples))

        self.num_samples_train = int(self.num_samples * ratio_train)
        self.num_samples_test = int(self.num_samples * ratio_test)
        self.idx_train = idx[:self.num_samples_train]
        self.idx_test = idx[self.num_samples_train: self.num_samples_train + self.num_samples_test]
        self.X, self.y = X, y
        self.X_train, self.y_train = X[self.idx_train], y[self.idx_train]
        self.X_test, self.y_test = X[self.idx_test], y[self.idx_test]

        if self.num_samples_test == 0:
            raise ValueError("Test set size is zero. Adjust your train/test split ratios.")

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

    @property
    def accuracy(self):
        preds = self.predict()
        count = 0
        for i, pred in enumerate(preds):
            if pred == self.y_test[i]: count += 1
        return count/self.num_samples

    def predict(self):
        return self.random_forest.predict(self.X_test)

    @staticmethod
    def _check_directory(directory_path):
        if not os.path.exists(directory_path):  # Check if the directory exists
            os.makedirs(directory_path)  # Create the directory
            Logger.info(f"Directory '{directory_path}' was created.")
        else:
            Logger.info(f"Directory '{directory_path}' already exists.")

    def test_regression(self, last_years_test = 1):
        plt.figure()
        plt.plot(self.y, '.-')
        plt.xlabel('day in 10 years'), plt.ylabel('min. daily temp')
        idx = last_years_test*365
        idx = min(idx, len(self.y_test))
        plt.figure()
        x = range(idx)
        ypred = self.predict()[:idx]
        ytest = self.y_test[:idx]
        for t, y1, y2 in zip(x, ytest, ypred):
            plt.plot([t, t], [y1, y2], 'k-')
        plt.plot([x[0], x[0]], [ytest[0], ypred[0]], 'k-', label='error')
        plt.plot(x, ytest, 'g.', label='test')
        plt.plot(x, ypred, 'y.', label='prediction')
        plt.xlabel('day in last {} years'.format(last_years_test))
        plt.ylabel('min. daily temperature')
        plt.legend()
        errors = ytest - ypred
        rmse = np.sqrt(np.mean(errors ** 2))
        plt.title('root mean square error : {:.3f}'.format(rmse))
        plt.show()

    def plot_accuracy(self):
        accuracy = self.evaluate
        Logger.info(f'{self._database_name} - Random forest ({str(self.random_forest.impurity)}) Accuracy: {np.mean(accuracy)}')
        plt.bar(range(1, len(accuracy) + 1, ), accuracy, color='blue')

        plt.title(f'{self._database_name} - Random Forest ({self.random_forest.impurity})')
        plt.xlabel('Tree')
        plt.ylabel('Accuracy')
        plt.ylim((0, 1.0))

        RandomForestEvaluator._check_directory('plots')

        plt.savefig(f'plots/{self._database_name}_{str(self.random_forest.impurity)}.png')