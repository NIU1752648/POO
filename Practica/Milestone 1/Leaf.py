from Node import Node

class Leaf(Node):
    def __init__(self, label):
        self._label = label

    @property
    def label(self):
        return self._label

    def predict(self, x):
        pass