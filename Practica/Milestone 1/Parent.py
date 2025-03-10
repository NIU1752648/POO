from Node import Node

class Parent(Node):
    def __init__(self, feature_index: int, threshold: float):
        self._feature_index = feature_index
        self._threshold = threshold

    @property
    def feature_index(self):
        return self._feature_index

    @property
    def threshold(self):
        return self._threshold

    def predict(self, x):
        pass