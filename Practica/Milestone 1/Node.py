from abc import ABC, abstractmethod

class Node(ABC):
    @abstractmethod
    def predict(self, x):
        """Predict must be implemented by all subclasses"""
        pass