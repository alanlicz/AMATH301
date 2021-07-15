import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import time
import mpmath
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize


#
# a = [2] * 1000
# b = [-1] * 999
# A = np.diag(a)
# A_1 = np.diag(b, 1)
# A_2 = np.diag(b, -1)
# A_3 = A + A_1 + A_2
#
# D = np.diag(np.diag(A_3))
# U = np.triu(A_3) - D
# L = np.tril(A_3) - D
#
#
# def max_eigenvalue(omega):
#     P = (1 / omega) * D + L
#     T = ((omega - 1) / omega) * D + U
#     M = -scipy.linalg.solve(P, T)
#     w, V = np.linalg.eig(M)
#     eig_max = np.max(np.abs(w))
#     return eig_max
#
#
# a = 1
# b = 2
# c = 0.5001
# x = c * a + (1 - c) * b
# fx = max_eigenvalue(x)
# y = (1 - c) * a + c * b
# fy = max_eigenvalue(y)
# tolerance = 1e-8
# t0 = time.time()
# for k in range(1000):
#     if fx < fy:
#         b = y
#         y = x
#         fy = fx
#         x = c * a + (1 - c) * b
#         fx = max_eigenvalue(x)
#     else:
#         a = x
#         x = y
#         fx = fy
#         y = (1 - c) * a + c * b
#         fy = max_eigenvalue(y)
#
#     if (b - a) < tolerance:
#         break
# t1 = time.time()
# print(t1 - t0)
#

# f = lambda x: np.sin(np.tan(x) - np.tan(np.sin(x)))
# a = 1.5646
# b = 1.5647
# tolerance = 1e-16
# c = (-1 + np.sqrt(5)) / 2
# x = c * a + (1 - c) * b
# fx = f(x)
# y = (1 - c) * a + c * b
# fy = f(y)
# count = 2
# for k in range(1000):
#     count += 1
#     if fx < fy:
#         b = y
#         y = x
#         fy = fx
#         x = c * a + (1 - c) * b
#         fx = f(x)
#     else:
#         a = x
#         x = y
#         fx = fy
#         y = (1 - c) * a + c * b
#         fy = f(y)
#
#     if (b - a) < tolerance:
#         break
# print(x)

# x0 = 1.5646745551757886
# k = 0
# X = np.zeros(1001)
# X[0] = x0
# tolerance = 1e-8
# f = lambda x: np.sin(np.tan(x) - np.tan(np.sin(x)))
# fprime = lambda x: mpmath.sec(x) ** 2 * np.cos(np.tan(x)) - np.cos(x) * mpmath.sec(np.sin(x)) ** 2
# fdprime = lambda x: mpmath.sec(x) ** 4 * (-np.sin(np.tan(x)) + mpmath.sec(np.sin(x)) ** 2) * \
#                     (np.sin(x) - 2 * np.cos(x) ** 2 * np.tan(np.sin(x)))
# for k in range(1000):
#     X[k + 1] = X[k] - fprime(X[k]) / fdprime(X[k])
#     if np.abs(fprime(X[k + 1])) < tolerance:
#         break
# X = X[:(k + 2)]
# guesses = X.size
# result = X[-1]
# print(X[-1], X[-2])

# x0 = 1.5
# k = 0
# X = np.zeros(1001)
# X[0] = x0
# tolerance = 1e-8
# f = lambda x: 2/3 * np.abs(x - 2) ** 2/3
# fprime = lambda x: (2/3)*(np.abs(x - 2)) ** (3/2)
# fdprime = lambda x: (x - 2) ** 2 / (2 * (np.abs(x - 2) ** 5 / 2))
# for k in range(1000):
#     X[k + 1] = X[k] - fprime(X[k]) / fdprime(X[k])
#     if np.abs(fprime(X[k + 1])) < tolerance:
#         break
# X = X[:(k + 2)]
# guesses = X.size
# print(X[-1])

f = lambda v: np.sin(v[0]) * np.e ** ((1 - np.cos(v[1])) ** 2) + \
              np.cos(v[1]) * np.e ** ((1 - np.sin(v[0])) ** 2) + (v[0] - v[1]) ** 2

x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.linspace(-2 * np.pi, 2 * np.pi, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
m = scipy.optimize.minimize(f, np.array([-1, -4]), method="Nelder-Mead").x

plt.plot(m[0], m[1], 'ro')
con = plt.contour(X, Y, Z, cmap=cm.rainbow, levels=50)

plt.clabel(con)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap=cm.plasma, alpha=0.8)
ax.view_init(22, 140)
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.plot(m[0], m[1], f(m), "ro")
plt.show()
