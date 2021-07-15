import numpy as np
import scipy.linalg

a = [2] * 114
b = [-1] * 113
A = np.diag(a)
A_1 = np.diag(b, 1)
A_2 = np.diag(b, -1)
A_3 = A + A_1 + A_2
A1 = A_3.copy()

rho = np.zeros((114, 1))
for j in range(114):
    rho[j, 0] = 2. * (1. - np.cos(53. * np.pi / 115)) * np.sin(53. * np.pi * (j+1) / 115)
A2 = rho.copy()


P = np.diag(np.diag(A1))
T = A1 - P
M = -scipy.linalg.solve(P, T)
w, V = np.linalg.eig(M)
A3 = np.max(np.abs(w))

tolerance = 1e-5
err = tolerance + 1
x0 = np.ones((114, 1))
X = np.zeros((114, 1))
X[:, 0:1] = x0

k = 0
if np.max(np.abs(w)) < 1:
    while err >= tolerance:
        X = np.hstack((X, scipy.linalg.solve_triangular(P, -T @ X[:, k:(k+1)] + A2)))
        err = np.max(np.abs(X[:, k+1] - X[:, k]))
        k = k + 1
A4 = X[:, k].reshape(114, 1)
A5 = k + 1

error_array = []
for i in range(114):
    error_array.append(A4[i, 0] - np.sin(53. * np.pi * (i + 1) / 115))
A6 = np.max(np.abs(error_array))

P2 = np.tril(A1)
T2 = A1 - P2
M2 = -scipy.linalg.solve(P2, T2)
w2, V2 = np.linalg.eig(M2)
A7 = np.max(np.abs(w2))
X2 = np.zeros((114, 1))
X2[:, 0:1] = x0
k2 = 0
err = tolerance + 1
if np.max(np.abs(w2)) < 1:
    while err >= tolerance:
        X2 = np.hstack((X2, scipy.linalg.solve_triangular(P2, -T2 @ X2[:, k2:(k2+1)] + A2, lower=True)))
        err = np.max(np.abs(X2[:, k2+1] - X2[:, k2]))
        k2 = k2 + 1
A8 = X2[:, k2].reshape(114, 1)
A9 = k2 + 1

error_array2 = []
for i in range(114):
    error_array2.append(A8[i, 0] - np.sin(53. * np.pi * (i + 1) / 115))
A10 = np.max(np.abs(error_array2))


D = np.diag(np.diag(A1))
U = np.triu(A1) - D
L = np.tril(A1) - D
P3 = (1 / 1.5) * D + L
A11 = P3.copy()
T3 = ((1.5 - 1) / 1.5) * D + U
M3 = -scipy.linalg.solve(P3, T3)
w3, V3 = np.linalg.eig(M3)
A12 = np.max(np.abs(w3))
X3 = np.zeros((114, 1))
X3[:, 0:1] = x0
k3 = 0
err = tolerance + 1
if np.max(np.abs(w3)) < 1:
    while err >= tolerance:
        X3 = np.hstack((X3, scipy.linalg.solve(P3, -T3 @ X3[:, k3:(k3+1)] + A2)))
        err = np.max(np.abs(X3[:, k3+1] - X3[:, k3]))
        k3 = k3 + 1
A13 = X3[:, k3].reshape(114, 1)
A14 = k3 + 1
error_array3 = []
for i in range(114):
    error_array3.append(A13[i, 0] - np.sin(53. * np.pi * (i + 1) / 115))
A15 = np.max(np.abs(error_array3))

