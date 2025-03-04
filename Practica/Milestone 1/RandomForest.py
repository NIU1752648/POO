"""
POO - Milestone 1
Random Forest Classifier
"""

from abc import ABC, abstractmethod

class Node(ABC):
    @abstractmethod
    def predict(self, x):
        """Predict must be implemented by all subclasses"""
        pass

class Leaf(Node):
    def __init__(self):
        pass

    def predict(self, x):
        pass

class Parent(Node):
    def __init__(self):
        pass

    def predict(self, x):
        pass

class RandomForest:
    pass