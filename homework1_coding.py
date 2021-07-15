import numpy as np
# Problem1
A = np.array([[1, -2.3], [4.5, np.e ** 2]])
B = np.array([[6, 2, 4, -3], [np.pi, 9, 3.6, -2.1]])
C = np.array([[3.7, -2.4, 0], [4, 1.8, -11.6], [2.3, -3.3, 5.9]])
x = np.array([[5], [np.sin(4)], [-3]])
y = np.array([8, -6])
z = np.array([[3], [0], [np.tan(2)], [-4.7]])

# Calculation
A1 = x * 3
A2 = np.transpose(z) @ np.transpose(B) + y
A3 = C @ x
A4 = A @ B
A5 = np.transpose(B) @ np.transpose(A)

# Problem 2
A6 = np.linspace(-4, 1, 73).reshape(1, -1)
A7 = np.cos(np.arange(0, 73)).reshape(1, -1)
A8 = A6 * A7
A9 = A6[:] / A7[:]
A10 = A6[:] ** 3 - A7[:]

# Problem 3
P1 = 3
for i in range(0, 3):
    P1 = 2.5 * P1 * (1 - P1 / 20)
A11 = P1

P2 = 8
for i in range(0, 4):
    P2 = 3.2 * P2 * (1 - P2 / 14)
A12 = P2

P3 = 5
for i in range(0, 3):
    P3 = P3 * np.e ** (2.6 * (1 - P3 / 12))
A13 = P3

P4 = 2
for i in range(0, 4):
    P4 = P4 * np.e ** (3 * (1 - P4 / 25))
A14 = P4

P5 = 0
for i in range(0, 500):
    P5 = P5 * np.e ** (3.1 * (1 - P5 / 20))
A15 = P5

