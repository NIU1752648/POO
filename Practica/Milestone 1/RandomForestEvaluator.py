from Dataset import Dataset
from RandomForestClassifier import RandomForestClassifier

class RandomForestEvaluator:
    def __init__(self, random_forest: RandomForestClassifier, test_dataset: Dataset):
        self._random_forest = random_forest
        self._test_dataset = test_dataset

    @property
    def random_forest(self):
        return self._random_forest

    @property
    def test_dataset(self):
        return self._test_dataset

    def evaluate(self):
        pass