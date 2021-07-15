import numpy as np
import scipy.integrate

T = 10
dt = 0.1

x_true = lambda t: 0.5 * (np.cos(t) + np.sin(t) + np.exp(-t))
f = lambda t, x: np.cos(t) - x

t = np.arange(0, T + dt, dt)
n = t.size
x_forward = np.zeros(n)
x_forward[0] = 1

for k in range(n - 1):
    x_forward[k + 1] = x_forward[k] + dt * f(t[k], x_forward[k])
A1 = np.reshape(x_forward, (1, 101))

error_forward = np.zeros(n)
for k in range(n):
    error_forward[k] = np.abs(x_forward[k] - x_true(t[k]))
A2 = np.reshape(error_forward, (1, 101))

x_backward = np.zeros(n)
x_backward[0] = 1
for k in range(n - 1):
    x_backward[k + 1] = (x_backward[k] + dt * np.cos(t[k + 1])) / (1 + dt)
A3 = np.reshape(x_backward, (1, 101))

error_backward = np.zeros(n)
for k in range(n):
    error_backward[k] = np.abs(x_backward[k] - x_true(t[k]))
A4 = np.reshape(error_backward, (1, 101))

tspan = (0, 10)
x0 = np.array([1])
sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)
x = sol.y[0, :]
A5 = np.reshape(x, (1, 101))

error_solve = np.zeros(n)
for k in range(n):
    error_solve[k] = np.abs(x[k] - x_true(t[k]))
A6 = np.reshape(error_solve, (1, 101))

# Problem 2
a = 8
T = 2
dt = 0.01

x_true = lambda t: 2 * np.arctan(np.exp(a * t) / (1 + np.sqrt(2)))
f = lambda t, x: a * np.sin(x)

t = np.arange(0, T + dt, dt)
n = t.size
x_forward = np.zeros(n)
x_forward[0] = np.pi / 4
for k in range(n - 1):
    x_forward[k + 1] = x_forward[k] + dt * f(t, x_forward[k])
A7 = np.reshape(x_forward, (1, 201))

error_forward = np.zeros(n)
for k in range(n):
    error_forward[k] = np.abs(x_forward[k] - x_true(t[k]))
error = np.reshape(error_forward, (1, 201))
A8 = np.max(error)

dt = 0.001
t = np.arange(0, T + dt, dt)
n = t.size
x_forward = np.zeros(n)
x_forward[0] = np.pi / 4
for k in range(n - 1):
    x_forward[k + 1] = x_forward[k] + dt * f(t, x_forward[k])

error_forward = np.zeros(n)
for k in range(n):
    error_forward[k] = np.abs(x_forward[k] - x_true(t[k]))
new_error = np.reshape(error_forward, (1, 2001))
new_error_max = np.max(new_error)
A9 = A8 / new_error_max

dt = 0.01
t = np.arange(0, T + dt, dt)
n = t.size

x_backward = np.zeros(n)
x_backward[0] = np.pi / 4
for k in range(n - 1):
    F = lambda z: z - x_backward[k] - a * dt * np.sin(z)
    z0 = 3
    z = scipy.optimize.fsolve(F, z0)
    x_backward[k + 1] = z
A10 = np.reshape(x_backward, (1, 201))

error_backward = np.zeros(n)
for k in range(n):
    error_backward[k] = np.abs(x_backward[k] - x_true(t[k]))
A11 = np.max(error_backward)

dt = 0.001
t = np.arange(0, T + dt, dt)
n = t.size

x_backward = np.zeros(n)
x_backward[0] = np.pi / 4
for k in range(n - 1):
    F = lambda z: z - x_backward[k] - a * dt * np.sin(z)
    z0 = 3
    z = scipy.optimize.fsolve(F, z0)
    x_backward[k + 1] = z

error_backward = np.zeros(n)
for k in range(n):
    error_backward[k] = np.abs(x_backward[k] - x_true(t[k]))
new_error_max = np.max(error_backward)
A12 = A11 / new_error_max

dt = 0.01
t = np.arange(0, T + dt, dt)
tspan = (0, 2)
n = t.size
x0 = np.array([np.pi / 4])
sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)
x = sol.y[0, :]
A13 = np.reshape(x, (1, 201))

error_solve = np.zeros(n)
for k in range(n):
    error_solve[k] = np.abs(x[k] - x_true(t[k]))
A14 = np.max(error_solve)

dt = 0.001
t = np.arange(0, T + dt, dt)
tspan = (0, 2)
n = t.size
x0 = np.array([np.pi / 4])
sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)
x = sol.y[0, :]

error_solve = np.zeros(n)
for k in range(n):
    error_solve[k] = np.abs(x[k] - x_true(t[k]))
new_error_max = np.max(error_solve)
A15 = A14 / new_error_max
print(A14)








