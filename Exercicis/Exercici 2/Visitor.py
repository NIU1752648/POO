from abc import ABC, abstractmethod
from Dual import Dual
import math

class Visitors:
    class Visitor(ABC):
        @abstractmethod
        def visit_dual(self, dual: Dual):
            pass

    class OperatorVisitor(ABC):
        @abstractmethod
        def visit_duals(self, a, b):
            pass

    # Define arithmetic operations
    class Addition(OperatorVisitor):
        def visit_duals(self, dual1: Dual, dual2: Dual):
            return Dual(dual1.a + dual2.a, dual1.b + dual2.b)

    class Subtraction(OperatorVisitor):
        def visit_duals(self, dual1: Dual, dual2: Dual):
            return Dual(dual1.a - dual2.a, dual1.b - dual2.b)

    class Multiplication(OperatorVisitor):
        def visit_duals(self, dual1: Dual, dual2: Dual):
            return Dual(dual1.a * dual2.a, (dual1.b * dual2.a) + (dual1.a * dual2.b))

    class Division(OperatorVisitor):
        def visit_duals(self, dual1: Dual, dual2: Dual):
            return Dual(dual1.a / dual2.a, (dual1.b * dual2.a - dual1.a * dual2.b) / (dual2.a ** 2))

    class Exponent(OperatorVisitor):
        def visit_duals(self, dual1: Dual, power: int):
            return Dual(dual1.a ** power, power * (dual1.a ** (power - 1)) * dual1.b)

    # Define elemental functions
    class Abs(Visitor):
        def visit_dual(self, dual: Dual):
            return Dual(abs(dual.a), dual.b * (dual.a / abs(dual.a)))

    class Sin(Visitor):
        def visit_dual(self, dual: Dual):
            return Dual(math.sin(dual.a), dual.b * math.cos(dual.a))

    class Cos(Visitor):
        def visit_dual(self, dual: Dual):
            return Dual(math.cos(dual.a), -dual.b * math.sin(dual.a))

    class Exp(Visitor):
        def visit_dual(self, dual: Dual):
            return Dual(math.exp(dual.a), dual.b * math.exp(dual.a))

    class Log(Visitor):
        def visit_dual(self, dual: Dual):
            return Dual(math.log(dual.a), dual.b/dual.a)
