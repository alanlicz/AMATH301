import numpy as np
import matplotlib.pyplot as plt
# x1 = 0
# x2 = 0
# x3 = 0
# x4 = 0
# for i in range(25000000):
#     x1 += 0.1
#
# for i in range(12500000):
#     x2 += 0.2
#
# for i in range(10000000):
#     x3 += 0.25
#
# for i in range(5000000):
#     x4 += 0.5
#
# y1 = abs(2500000 - x1)
# y2 = abs(2500000 - x2)
# y3 = abs(2500000 - x3)
# y4 = abs(2500000 - x4)

# y1 = []
# y2 = []
# y3 = []
# y4 = []
# result = 0
# for i in range(1001):
#     result = 2 * result * (1 - result / 30)
#     y1.append(result)
#
# result = 15
# for i in range(1001):
#     result = 2 * result * (1 - result / 30)
#     y2.append(result)
#
# result = 3
# for i in range(1001):
#     result = 2 * result * (1 - result / 30)
#     y3.append(result)
#
# result = 25
# for i in range(1001):
#     result = 2 * result * (1 - result / 30)
#     y4.append(result)
#
# x = np.arange(1001)
# plt.plot(x, y1, "k")
# plt.plot(x, y2, "y")
# plt.plot(x, y3, "r")
# plt.plot(x, y4, "b")
# plt.legend(("P(0) = 0", "P(0) = 15", "P(0) = 3", "P(0) = 25"),
#            loc="center right")
# plt.title("Population growth")
# plt.xlabel("Time")
# plt.ylabel("Population")
# plt.show()


def logistic_map(r, initial, k, n):
    result = initial
    p_array = np.zeros(n)
    for i in range(n):
        result = r * result * (1 - result / k)
        p_array[i] = result
    return p_array
#
#
# r_value = np.arange(1.1, 3, 0.1)
# equilibrium = []
# for n_value in range(r_value.size):
#     equilibrium_array = logistic_map(r_value[n_value], 1, 30, 1000)
#     equilibrium.append(equilibrium_array[-1])
# plt.plot(r_value, equilibrium, "k")
#
#
# plt.title("Equilibrium vs r")
# plt.xlabel("r")
# plt.ylabel("Population")
# plt.show()


r_value = np.arange(3.01, 3.44949, 0.01)
equilibrium = []
equilibrium2 = []
for n_value in range(r_value.size):
    equilibrium_array = logistic_map(r_value[n_value], 1, 30, 1000)
    equilibrium.append(equilibrium_array[-1])
    equilibrium2.append(equilibrium_array[-2])
plt.plot(r_value, equilibrium, "k")
plt.plot(r_value, equilibrium2, "r")
plt.xlabel("r")
plt.ylabel("Population")
plt.title("Equilibrium vs r")
plt.legend(("P equilibrium at P(1000)", "p equilibrium at P(999)"), loc="best")
plt.show()

