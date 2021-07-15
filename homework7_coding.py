import numpy as np
import scipy.integrate

data = np.genfromtxt('population.csv', delimiter=',')
t = data[0, :]
N = data[1, :]

dx = 10
A1 = (3 * N[-1] - 4 * N[-2] + N[-3]) / (2 * dx)
A2 = (N[10] - N[8]) / (2 * dx)
A3 = (-3 * N[0] + 4 * N[1] - N[2]) / (2 * dx)
A4 = np.zeros(24)
A4[-1] = A1
A4[0] = A3


for i in range(1, 23):
    A4[i] = (N[i+1] - N[i-1]) / (2 * dx)
A4 = A4.reshape(1, 24)
print(A1, A2, A3, A4)
A5 = np.zeros(24)
for i in range(0, 24):
    A5[i] = A4[0, i] / N[i]
A5 = np.reshape(A5, (1, 24))
A6 = np.zeros(24)
total = 0
for i in range(A5.size):
    total += A5[0, i]
A6 = total / 24
data = np.genfromtxt('brake_pad.csv', delimiter=',')
r = data[0, :]
T = data[1, :]
dx_2 = 0.017
A7 = 0
for i in range(10):
    A7 += dx_2 * 0.7051 * r[i] * T[i]
A = 0
for i in range(10):
    A += dx_2 * 0.7051 * r[i]
A8 = A7 / A

A9 = 0
for i in range(1, 11):
    A9 += dx_2 * 0.7051 * r[i] * T[i]
A_2 = 0
for i in range(1, 11):
    A_2 += dx_2 * 0.7051 * r[i]
A10 = A9 / A_2

A11 = 0
for i in range(10):
    A11 += ((dx_2 * 0.7051 * r[i] * T[i]) + (dx_2 * 0.7051 * r[i+1] * T[i+1])) / 2
A_3 = 0
for i in range(10):
    A_3 += ((dx_2 * 0.7051 * r[i]) + (dx_2 * 0.7051 * r[i+1])) / 2
A12 = A11 / A_3



F = lambda x: x**2/2-x**3/3
µ = 0.95
F1 = lambda x: µ/np.sqrt(F(µ)-F(µ * x))
Int, err = scipy.integrate.quad(F1, 0, 1)
A13 = Int

µ = 0.5
F1 = lambda x: µ/np.sqrt(F(µ)-F(µ * x))
Int, err = scipy.integrate.quad(F1, 0, 1)
A14 = Int

µ = 0.01
F1 = lambda x: µ/np.sqrt(F(µ)-F(µ * x))
Int, err = scipy.integrate.quad(F1, 0, 1)
A15 = Int


