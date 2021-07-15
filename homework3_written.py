import numpy as np
import scipy.linalg
import time
#
# a = 1e21 - (1e21 - 1e5)
# b = (1e21 - 1e21) + 1e5
#
#
# A = np.array([[1e-20, 1], [1, 1]])
#
# L = np.array([[1, 0], [1e20, 1]])
# U = np.array([[1e-20, 1], [0, 1 - 1e20]])
#
#
# B = np.array([[1, 1], [1e-20, 1]])
# L_2 = np.array([[1, 0], [1e-20, 1]])
# U_2 = np.array([[1, 1], [0, 1 - 1e-20]])
#
#
# P_3, L_3, U_3 = scipy.linalg.lu(A)
# P_4, L_4, U_4 = scipy.linalg.lu(B)
# print(L_4)
# print(U_4)

A = np.genfromtxt('example_matrix.csv', delimiter=',')
print("This is G.E:")
t0 = time.time()
r_sum = 0.0
for n in range(100):
    b = np.random.rand(3000, 1)
    x = scipy.linalg.solve(A, b)
    r = A @ x - b
r = np.max(np.abs(r))
r_sum += r
average_r = r_sum / 100
t1 = time.time()
print(t1 - t0)
print(average_r)
print()

print("This is LU:")
t0 = time.time()
r_sum = 0.0
P, L, U = scipy.linalg.lu(A)
P = P.T
for n in range(100):
    b = np.random.rand(3000, 1)
    y = scipy.linalg.solve_triangular(L, P @ b, lower=True)
    x = scipy.linalg.solve_triangular(U, y)
    r = A @ x - b
    r = np.max(np.abs(r))
    r_sum += r
average_r = r_sum / 100
t1 = time.time()
print(t1 - t0)
print(average_r)
print()

print("This is inverse:")
t0 = time.time()
r_sum = 0.0
A_reverse = scipy.linalg.inv(A)
for n in range(100):
    b = np.random.rand(3000, 1)
    x = A_reverse @ b
    r = A @ x - b
    r = np.max(np.abs(r))
    r_sum += r
average_r = r_sum / 100
t1 = time.time()
print(t1 - t0)
print(average_r)
print()
