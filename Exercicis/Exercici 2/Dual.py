from abc import ABC, abstractmethod
import math

class Visitor(ABC):
    @abstractmethod
    def visit_dual(self, dual):
        pass

class OperatorVisitor(ABC):
    @abstractmethod
    def visit_duals(self, a, b):
        pass

class Dual:
    def __init__(self, a: float, b: float):
        self._a = a
        self._b = b

    @property
    def a(self) -> float:
        return self._a

    @property
    def b(self) -> float:
        return self._b

    def __str__(self):
        return f"<{self.a, self.b}>"

    def __add__(self, other):
        return self.operate_two(other, Addition())

    def __sub__(self, other):
        return self.operate_two(other, Subtraction())

    def __mul__(self, other):
        return self.operate_two(other, Multiplication())

    def __div__(self, other):
        return self.operate_two(other, Division())

    def __pow__(self, power: int):
        if self.a == 0: raise ArithmeticError("Cannot raise a null Dual to any power.")
        return self.operate_two(power, Exponent())

    def __abs__(self):
        if self.a == 0: raise ArithmeticError("Cannot evaluate the absolute value of a null Dual.")
        return self.operate(Abs())

    def sin(self):
        return self.operate(Sin())

    def cos(self):
        return self.operate(Cos())

    def exp(self):
        return self.operate(Exp())

    def log(self):
        return self.operate(Log())

    def operate_two(self, other, visitor: OperatorVisitor):
        return visitor.visit_duals(self, other)

    def operate(self, visitor: Visitor):
        return visitor.visit_dual(self)

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
