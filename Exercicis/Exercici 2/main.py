from Dual import Dual
import matplotlib.pyplot as plt

if __name__ == "__main__":
    def f(x: Dual) -> Dual:
        return x ** 2 + x.sin()

    f_s = [f(Dual(i, 1)) for i in range(1, 101)]

    x = range(1, 101)
    y = [f.a for f in f_s]
    y_primes = [f.b for f in f_s]

    plt.plot(x, y, label="f(x)")
    plt.plot(x, y_primes, label="f'(x)")
    plt.show()