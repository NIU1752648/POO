import Dual
import numpy as np

class DualMulti:
    def __init__(self, a: np.array, b: np.array):
        self._a = a
        self._b = b

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def __str__(self):
        r = ""
        for i in range(self.a):
            r += f"<{self.a[i]} {self.b[i]} "
        return r
