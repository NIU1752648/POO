from abc import ABC, abstractmethod

class Node(ABC):
    @abstractmethod
    def __init__(self):
        """Constructor must be implemented by all subclasses"""
        pass

    @abstractmethod
    def predict(self, x):
        """Predict must be implemented by all subclasses"""
        pass