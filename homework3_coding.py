import numpy as np
import scipy.linalg

A = np.genfromtxt('bridge_matrix.csv', delimiter=',')
b = [[0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 8, 0, 4]]

b = np.transpose(b)
A1 = scipy.linalg.solve(A, b)
P, L, U = scipy.linalg.lu(A)
P = P.T
A2 = L.copy()
A3 = scipy.linalg.solve_triangular(L, P @ b, lower=True)
b = b.astype('float64')

A4 = 0
A5 = 0
flag = True
while flag:
    b[8, 0] += 0.01
    y = scipy.linalg.solve_triangular(L, P @ b, lower=True)
    f = scipy.linalg.solve_triangular(U, y)
    for n in range(13):
        if abs(f[n, 0]) > 30:
            A4 = b[8, 0]
            A5 = n + 1
            flag = False

A_2 = np.array([[1.003, -0.05], [0.05, 1.003]])
P_2, L_2, U_2 = scipy.linalg.lu(A_2)
P_2 = P_2.T
result = np.zeros(shape=(2, 1001))
result[:, 0] = [1, -1]
temp = [1, -1]
temp = np.transpose(temp)
for i in range(1, 1001):
    y_2 = scipy.linalg.solve_triangular(L_2, P_2 @ temp, lower=True)
    temp = scipy.linalg.solve_triangular(U_2, y_2)
    result[:, i] = temp
A6 = np.zeros(shape=(1, 1001))
for i in range(1001):
    A6[0, i] = result[0, i]
A7 = np.zeros(shape=(1, 1001))
for i in range(1001):
    A7[0, i] = result[1, i]

A8 = np.zeros(shape=(1, 1001))
for i in range(1001):
    A8[0, i] = np.sqrt(A6[0, i] ** 2 + A7[0, i] ** 2)

A9 = 0
A10 = 0
for i in range(1001):
    if A8[0, i] < 0.05:
        A9 = i
        A10 = A8[0, i]
        break


def y_rotation(theta):
    r = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return r


A11 = y_rotation(np.pi / 10)
x = [[0.3, 1.4, -2.7]]
x = np.transpose(x)
A12 = y_rotation(3 * np.pi / 8) @ x
y = [[1.2, -0.3, 2]]
y = np.transpose(y)
degree = y_rotation(np.pi / 7)
A13 = scipy.linalg.solve(degree, y)
A14 = scipy.linalg.inv(y_rotation(np.pi * 5 / 7))
A15 = - 5 * np.pi / 7

