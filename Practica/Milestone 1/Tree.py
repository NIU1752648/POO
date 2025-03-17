from abc import ABC, abstractmethod
from collections import deque

class Tree:
    class Node(ABC):
        @abstractmethod
        def __init__(self):
            """Constructor must be implemented by all subclasses"""
            pass

        @abstractmethod
        def predict(self, x):
            """Predict must be implemented by all subclasses"""
            pass

    class Leaf(Node):
        def __init__(self, label):
            self._label = label

        @property
        def label(self):
            return self._label

        def predict(self, x):
            return self.label

    class Parent(Node):
        def __init__(self, feature_index: int, threshold: float):
            self._feature_index = feature_index
            self._threshold = threshold
            self.left_child = None
            self.right_child = None

        @property
        def feature_index(self):
            return self._feature_index

        @property
        def threshold(self):
            return self._threshold

        def predict(self, x):
            return x[self._feature_index] < self._threshold

        def __iter__(self):
            queue = deque([self])  # Initialize the queue with the root node
            while queue:
                node = queue.popleft()  # Get the next node
                yield node # Yield the current node's value
                if isinstance(node, Tree.Parent):
                    if node.left_child is not Tree.Leaf: # Add children to the queue
                        queue.append(node.left_child)
                    if node.right_child is not Tree.Leaf:
                        queue.append(node.right_child)
