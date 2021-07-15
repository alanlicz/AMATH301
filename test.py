import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

T = 200
a = 8

f = lambda t, x: a * np.sin(x)
x_true = lambda t: 2 * np.arctan(np.exp(a*t) / (1 + 2**0.5))

# tplot = np.linspace(0, 200, 1000)
# x = x_true(tplot)
#
# plt.plot(tplot, x_true(tplot), 'k')
# plt.xlabel("t")
# plt.ylabel("x")
# plt.title("x vs t, the real solution")
# plt.show()


# def backward(dt):
#     t = np.arange(0, T + dt, dt)
#     n = t.size
#     x = np.zeros(n)
#     x[0] = np.pi/4
#     for k in range(n - 1):
#         F = lambda z: z - x[k] - a * dt * np.sin(z)
#         z0 = 3
#         x[k+1] = scipy.optimize.fsolve(F, z0)
#
#     err_global = np.abs(x[-1] - x_true(t[-1]))
#     print("The global error is ", err_global)
#     max_error = np.max(np.abs(x - x_true(t)))
#     print("The maximum error is ", max_error)
#
#     plt.plot(tplot, x_true(tplot), 'k')
#     plt.plot(t, x, 'b')
#     plt.xlabel("t")
#     plt.ylabel("x")
#     plt.title("x vs t")
#     plt.legend(("real solution", "approximation"), loc="best")
#     plt.show()
#
#
# backward(0.01)
# backward(0.625)
# backward(150)
# backward(0.001)

def forward(dt):
    t = np.arange(0, T + dt, dt)
    n = t.size
    x = np.zeros(n)
    x[0] = np.pi/4
    for k in range(n - 1):
        x[k + 1] = x[k] + dt * f(t[k], x[k])

    err_global = np.abs(x[-1] - x_true(t[-1]))
    print("The global error is ", err_global)
    max_error = np.max(np.abs(x - x_true(t)))
    print("The maximum error is ", max_error)

    # plt.plot(tplot, x_true(tplot), 'k')
    # plt.plot(t, x, 'b')
    # plt.xlabel("t")
    # plt.ylabel("x")
    # plt.title("x vs t")
    # plt.legend(("real solution", "approximation"), loc="best")
    # plt.show()


# forward(0.01)
hw8_written2(1).py
forward(0.1)
# forward(0.2)
# forward(0.3)
