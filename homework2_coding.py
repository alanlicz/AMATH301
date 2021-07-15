import numpy as np

A = np.empty((19, 34))
for i in range(19):
    for j in range(34):
        A[i, j] = 1 / ((i + 1) * (j + 1))
A1 = A.copy()

B = A.copy()
B[16, :] = 0
B[:, 32] = 0
A2 = B

A3 = B[14:19, 31:34]

A4 = B[:, 4].reshape(19, 1)

A5 = 0
for i in range(1, 11):
    A5 += 1 / i

A6 = 0
for i in range(1, 101):
    A6 += 1 / i

A8 = 0
for i in range(1, 101):
    A8 += 1 / i
    if A8 > 5:
        A7 = i
        break

A10 = 0
for i in range(1, 2000000):
    A10 += 1 / i
    if A10 > 15:
        A9 = i
        break


def logistic_map(r, initial, k, n):
    global i
    result = initial
    for i in range(n):
        result = r * result * (1 - result / k)
    return result


A11_1 = logistic_map(2.5, 3, 20, 998)
A11_2 = logistic_map(2.5, 3, 20, 999)
A11_3 = logistic_map(2.5, 3, 20, 1000)
A11 = np.array([[A11_1, A11_2, A11_3]])

A12_1 = logistic_map(3.2, 3, 20, 998)
A12_2 = logistic_map(3.2, 3, 20, 999)
A12_3 = logistic_map(3.2, 3, 20, 1000)
A12 = np.array([[A12_1, A12_2, A12_3]])

A13_1 = logistic_map(3.5, 3, 20, 998)
A13_2 = logistic_map(3.5, 3, 20, 999)
A13_3 = logistic_map(3.5, 3, 20, 1000)
A13 = np.array([[A13_1, A13_2, A13_3]])

