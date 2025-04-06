from Dual import Dual
import matplotlib.pyplot as plt
import numpy as np

def f(x: Dual) -> Dual:
    return (x ** 2) * (x.sin() ** 2 + x.cos())

def gradient_ascent():
    h = 1e-3
    K = 1000
    values = []
    for x_old in np.linspace(0, 2 * np.pi, 10):
        x_new = x_old
        for _ in range(K):
            x_new += h * f(Dual(x_new, 1)).b
        values.append(x_new)
    return values

if __name__ == "__main__":
    x = gradient_ascent()
    print(x)