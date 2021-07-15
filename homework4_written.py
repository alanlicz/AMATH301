import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
#
# A = np.genfromtxt('hw4_matrix.csv', delimiter=',')
# b = np.random.rand(20, 1)
# x = scipy.linalg.solve(A, b)
# print(x)
#
# print("a) Gaussian method find error:")
# P = np.diag(np.diag(A))
# T = A - P
# M = -scipy.linalg.solve(P, T)
# w, V = np.linalg.eig(M)
# tolerance = 1e-4
# x0 = np.ones((20, 1))
# X = np.zeros((20, 100001))
# error1 = np.zeros(100000)
# X[:, 0:1] = x0
# error1[0] = np.max(np.abs(X[:, 0:1] - x))
# k = 0
# for k in range(100000):
#     X[:, (k+1):(k+2)] = scipy.linalg.solve_triangular(P, -T @ X[:, k:(k+1)] + b)
#     error1[k+1] = np.max(np.abs(X[:, (k+1):(k+2)] - x))
#     if np.max(np.abs(X[:, k+1] - X[:, k])) < tolerance:
#         break
#
# X = X[:, :(k+2)]
# error1 = error1[:k+2]
# print(error1)
#
#
# print("b) Guass-Seidel method find error:")
# P = np.tril(A)  # set P as lower triangle
# T = A - P
# M = -scipy.linalg.solve(P, T)
# w, V = np.linalg.eig(M)
# tolerance = 1e-4
# x0 = np.ones((20, 1))
# X = np.zeros((20, 100001))
# error2 = np.zeros(100000)
# X[:, 0:1] = x0
# error2[0] = np.max(np.abs(X[:, 0:1] - x))
# k = 0
# for k in range(100000):
#     X[:, (k+1):(k+2)] = scipy.linalg.solve_triangular(P, -T @ X[:, k:(k+1)] + b, lower=True)
#     error2[k+1] = np.max(np.abs(X[:, (k+1):(k+2)] - x))
#     if np.max(np.abs(X[:, k+1] - X[:, k])) < tolerance:
#         break
#
# X = X[:, :(k+2)]
# error2 = error2[:k+2]
# print(error2)
#
# x_1 = np.arange(1, error1.size + 1)
# x_2 = np.arange(1, error2.size + 1)
# y_1 = error1
# y_2 = error2
# plt.title("Error of two methods")
# plt.xlabel("k")
# plt.ylabel("error")
# plt.semilogy(x_1, y_1, 'g')
# plt.semilogy(x_2, y_2, 'y')
# plt.legend(("Jacobi Method", "Gauss-Seidal Method"), loc="upper center")
# plt.show()

a = [2] * 114
b = [-1] * 113
A = np.diag(a)
A_1 = np.diag(b, 1)
A_2 = np.diag(b, -1)
A_3 = A + A_1 + A_2

D = np.diag(np.diag(A_3))
U = np.triu(A_3) - D
L = np.tril(A_3) - D

# c = 1
# eig_array = []
# omega_array = np.arange(1, 2, 0.001)
# for i in range(1000):
#     P = (1 / c) * D + L
#     T = ((c - 1) / c) * D + U
#     M = -scipy.linalg.solve(P, T)
#     w, V = np.linalg.eig(M)
#     eig_max = np.max(np.abs(w))
#     eig_array.append(eig_max)
#     c += 0.001
# print(eig_array)
# plt.title("Absolute value of the largest eigenvalue of M vs Omega")
# plt.xlabel("Omega")
# plt.ylabel("Eigen value")
# plt.plot(omega_array, eig_array, 'y')
# plt.show()
# omega_min_index = np.argmin(eig_array)
# print(omega_array[omega_min_index])

tolerance = 1e-5
err = tolerance + 1
x0 = np.ones((114, 1))
X = np.zeros((114, 1))
X[:, 0:1] = x0

P = (1 / 1.9469999999998957) * D + L
T = ((1.9469999999998957 - 1) / 1.9469999999998957) * D + U

k = 0
M = -scipy.linalg.solve(P, T)
w, V = np.linalg.eig(M)
if np.max(np.abs(w)) < 1:
    while err >= tolerance:
        X = np.hstack((X, scipy.linalg.solve(P, -T @ X[:, k:(k+1)])))
        err = np.max(np.abs(X[:, k+1] - X[:, k]))
        k = k + 1
print(k)
