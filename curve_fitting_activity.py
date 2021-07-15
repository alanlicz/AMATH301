import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize

# Load data from fitdata.csv
data = np.genfromtxt('activity_data2.csv', delimiter=',')
x = data[0, :]
y = data[1, :]
n = x.size

# Plot the data
plt.plot(x, y, 'ko')

# Guess a quadratic fit of the form f(x) = ax^2 + bx + c
miu = 2.12047639
sigma2 = 1.11113674
f = lambda x_: (1 / math.sqrt(2 * np.pi * sigma2)) * np.exp(-(x_ - miu) ** 2 / 2 * sigma2)

# Plot the quadratic
xplot = np.linspace(-6, 8, 1000)
yplot = f(xplot)
plt.plot(xplot, yplot)

for k in range(n):
    xdata = x[k]
    ydata = y[k]
    yhat = f(xdata)

    # Plot a point at the predicted value
    plt.plot(xdata, yhat, 'ko')
    # Plot a red line between the data point and the predicted value
    plt.plot([xdata, xdata], [ydata, yhat], 'r')

# Choose a reasonable scale for the y-axis
plt.ylim([-0.5, 1])
plt.show()


def rms_error(miu_, sigma2_, x_, y_):
    total_sum = 0
    for k2 in range(n):
        result = ((1 / math.sqrt(2 * np.pi * sigma2_)) * np.exp(-(x_[k2] - miu_) ** 2 / 2 * sigma2_) - y_[k2]) ** 2
        total_sum += result
    error = math.sqrt(total_sum / n)
    return error


def average_error(miu_, sigma2_, x_, y_):
    total_sum = 0
    for k2 in range(n):
        result = np.abs(((1 / math.sqrt(2 * np.pi * sigma2_)) * np.exp(-(x_[k2] - miu_) ** 2 / 2 * sigma2_) -
                         y_[k2]))
        total_sum += result
    error = total_sum / n
    return error


def maximum_error(miu_, sigma2_, x_, y_):
    predicted = (1 / math.sqrt(2 * np.pi * sigma2_)) * np.exp(-(x_ - miu_) ** 2 / 2 * sigma2_)
    return np.linalg.norm(predicted - y_, np.inf)


rms_wrapper = lambda coeffs: rms_error(coeffs[0], coeffs[1], x, y)
average_wrapper = lambda coeffs: average_error(coeffs[0], coeffs[1], x, y)
maximum_wrapper = lambda coeffs: maximum_error(coeffs[0], coeffs[1], x, y)
print(rms_wrapper(np.array([0, 1])))
print(scipy.optimize.minimize(rms_wrapper, np.array([2, 1]), method="Nelder-Mead").x)
print(average_error(0, 1, x, y))
print(average_wrapper(np.array([0, 1])))
print(scipy.optimize.minimize(average_wrapper, np.array([2, 1]), method="Nelder-Mead").x)
print(maximum_error(0, 1, x, y))
print(maximum_wrapper(np.array([0, 1])))
print(scipy.optimize.minimize(maximum_wrapper, np.array([2, 1]), method="Nelder-Mead").x)
