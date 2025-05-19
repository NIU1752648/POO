from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque

class Tree:
    class Node(ABC):
        @abstractmethod
        def __init__(self):
            pass

        @abstractmethod
        def predict(self, x):
            pass

        @abstractmethod
        def accept_visitor(self, v: Visitor):
            pass

    class Leaf(Node):
        def __init__(self, label):
            self._label = label

        @property
        def label(self):
            return self._label

        def predict(self, x):
            return self.label

        def accept_visitor(self, v: Visitor):
            v.visitLeaf(self)

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

        def accept_visitor(self, v: Visitor):
            v.visitParent(self)

    class Visitor(ABC):
        @abstractmethod
        def visitParent(self, p: Parent):
            pass

        @abstractmethod
        def visitLeaf(self, p: Leaf):
            pass

class Visitors:
    class FeatureImportance(Tree.Visitor):
        # Counts the number of times each feature is used in the tree
        def __init__(self):
            self._occurrences = dict()

        @property
        def occurrences(self):
            return self._occurrences

        def visitParent(self, p: Tree.Parent):
            k = p.feature_index
            if k in self._occurrences.keys():
                self._occurrences[k] += 1
            else:
                self._occurrences[k] = 1

            p.left_child.accept_visitor(self)
            p.right_child.accept_visitor(self)

        def visitLeaf(self, p: Tree.Leaf):
            pass

        def print_importance(self):
            print(self._occurrences)

    class PrinterTree(Tree.Visitor):
        # Prints the tree in a readable format
        def __init__(self):
            self._depth = 0

        def visitParent(self, p: Tree.Parent):
            print('     ' * self._depth + f'parent, {p.feature_index}, {p.threshold}')
            self._depth += 1
            p.left_child.accept_visitor(self)
            p.right_child.accept_visitor(self)
            self._depth -= 1

        def visitLeaf(self, p: Tree.Leaf):
            print('     ' * self._depth + f'leaf, {p.label}')