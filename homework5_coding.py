import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from math import e

tolerance = 1e-8


def find_minimum_power_function(power):
    x0 = 2
    k = 0
    X = np.zeros(10001)
    X[0] = x0
    f = lambda x: x ** power
    fprime = lambda x: power * (x ** (power - 1))
    fdprime = lambda x: power * (power - 1) * (x ** (power - 2))
    for k in range(10000):
        X[k + 1] = X[k] - fprime(X[k]) / fdprime(X[k])
        if np.abs(fprime(X[k + 1])) < tolerance:
            break
    X = X[:(k + 2)]
    guesses = X.size
    result = X[-1]
    return guesses, result


A1, A2 = find_minimum_power_function(2)
A3, A4 = find_minimum_power_function(500)
A5, A6 = find_minimum_power_function(1000)

f2 = lambda x: x ** 1000
fprime2 = lambda x: 1000 * (x ** 999)
fdprime2 = lambda x: 999000 * (x ** 998)

a = -2
b = 2
c = (-1 + np.sqrt(5)) / 2

x = c * a + (1 - c) * b
fx = f2(x)
y = (1 - c) * a + c * b
fy = f2(y)
count = 2
for k in range(1000):
    count += 1
    if fx < fy:
        b = y
        y = x
        fy = fx
        x = c * a + (1 - c) * b
        fx = f2(x)
    else:
        a = x
        x = y
        fx = fy
        y = (1 - c) * a + c * b
        fy = f2(y)

    if (b - a) < tolerance:
        break
A7 = count
A8 = x


c = lambda t: 1.3 * (e ** (- t / 11) - e ** (-4 * t / 3))
c_inverse = lambda t: - 1.3 * (e ** (- t / 11) - e ** (-4 * t / 3))
A9 = minimize_scalar(c_inverse, bounds=(0, 5), method="Bounded").x
A10 = c(A9)

h = lambda x: - 1 / ((x - 0.3) ** 2 + 0.01) - 1 / ((x - 0.9) ** 2 + 0.04) - 6
h_inverse = lambda x:  1 / ((x - 0.3) ** 2 + 0.01) + 1 / ((x - 0.9) ** 2 + 0.04) + 6
A11 = minimize_scalar(h, bounds=(0, 0.5), method="Bounded").x
A12 = minimize_scalar(h_inverse, bounds=(0.5, 0.8), method="Bounded").x
A13 = minimize_scalar(h, bounds=(0.8, 2), method="Bounded").x

f3 = lambda v: (v[0] ** 2 + v[1] - 11) ** 2 + (v[0] + v[1] ** 2 - 7) ** 2
f3_inversed = lambda v: - (v[0] ** 2 + v[1] - 11) ** 2 - (v[0] + v[1] ** 2 - 7) ** 2
min1 = minimize(f3, np.array([-2, 3]), method="Nelder-Mead")
A14 = min1.x.reshape(2, 1)
min2 = minimize(f3_inversed, np.array([0, 0]), method="Nelder-Mead")
A15 = min2.x.reshape(2, 1)

