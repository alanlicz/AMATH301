import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
# lam = -2100
# f = lambda t, x: lam * (x - np.cos(t)) - np.sin(t)
# T = 2
# error = np.zeros(21)
# dt = 0.00094
# for i in range(21):
#     t = np.arange(0, T + dt, dt)
#     n = t.size
#     x_forward = np.zeros(n)
#     x_forward[0] = 1
#     for j in range(n - 1):
#         x_forward[j + 1] = x_forward[j] + dt * f(t[j], x_forward[j])
#     error[i] = np.abs(x_forward[-1] - np.cos(2))
#     dt += 0.000001
# dt_array = np.linspace(9.4e-4, 9.6e-4, 21)
# plt.plot(dt_array, error, "b")
# plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
# plt.xlabel("dx value")
# plt.ylabel("error")
# plt.title("error vs different dx value")
# plt.show()

a = 8
f = lambda t, x: a * np.sin(x)
# x0 = np.pi / 4
x_true = lambda t: 2 * np.arctan(np.exp(a * t) / (1 + np.sqrt(2)))
T = 200
tplot = np.linspace(0, 200, 1000)
# plt.plot(tplot, x_true(tplot), 'k')
# plt.show()

dt = 0.625
t = np.arange(0, T + dt, dt)
n = t.size
# x_backward = np.zeros(n)
# x_backward[0] = np.pi / 4
x_forward = np.zeros(n)
x_forward[0] = np.pi / 4
# for k in range(n - 1):
#     F = lambda z: z - x_backward[k] - a * dt * np.sin(z)
#     z0 = 3
#     z = scipy.optimize.fsolve(F, z0)
#     x_backward[k + 1] = z
# max_error = np.max(np.abs(x_backward - x_true(t)))
# global_error = np.max(np.abs(x_backward[-1] - x_true(t[-1])))

for k in range(n - 1):
    x_forward[k + 1] = x_forward[k] + dt * f(t[k], x_forward[k])
max_error_2 = np.max(np.abs(x_forward - x_true(t)))
global_error_2 = np.abs(x_forward[-1] - x_true(t[-1]))

# tplot = np.arange(0, T + dt, dt)
# plt.plot(tplot, x_forward, "r")
# plt.plot(tplot, x_true(t), "b")

plt.show()
print(max_error_2, global_error_2)

