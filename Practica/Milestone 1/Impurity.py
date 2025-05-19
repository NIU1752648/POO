from abc import ABC, abstractmethod
import numpy as np
from math import log

from Dataset import Dataset

class Impurities:
    class Impurity(ABC):
        @abstractmethod
        def impurity(self, dataset: Dataset):
            pass

    class GiniIndex(Impurity):
        def impurity(self, dataset: Dataset):
            # the Gini impurity equation
            gini = 1
            for label in np.unique(dataset.y):
                count = 0
                for elem in dataset.y:
                    if elem == label:
                        count += 1
                gini -= (count / dataset.num_samples) ** 2
            return gini
        def __str__(self):
            return 'Gini'

    class Entropy(Impurity):
        def impurity(self, dataset: Dataset):
            # the entropy impurity equation
            entropy = 0
            for label in np.unique(dataset.y):
                count = 0
                for elem in dataset.y:
                    if elem == label:
                        count += 1
                entropy -= (count / dataset.num_samples) * log(count / dataset.num_samples)
            return entropy

        def __str__(self):
            return 'Entropy'   

    class SumSquareError(Impurity):
        # Impurity measure for regression tasks (sum of squared errors).
        def impurity(self, dataset) -> float:
            return np.sum((dataset.y - dataset.mean_value()) ** 2)
        
        def __str__(self):
            return 'Sum of Square Errors'
