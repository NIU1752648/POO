from abc import ABC, abstractmethod

class Impurities:
    class Impurity(ABC):
        @abstractmethod
        def impurity(self):
            pass

    class GiniIndex(Impurity):
        def impurity(self):
            pass

    class Entropy(Impurity):
        def impurity(self):
            pass