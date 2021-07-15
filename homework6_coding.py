import numpy as np
import scipy.optimize
import scipy.interpolate

data = np.genfromtxt('CO2_data.csv', delimiter=',')
t = data[0, :]
co2 = data[1, :]
n = t.size

A1_temp = np.polyfit(t, co2, 1)
A1 = np.reshape(A1_temp, (1, 2))
total = 0
for k in range(n):
    result = ((t[k] * A1_temp[0] + A1_temp[1]) - co2[k]) ** 2
    total += result
A2 = np.sqrt(total / n)


def exponential_function(a, r, b):
    total2 = 0
    for k2 in range(n):
        predicted2 = a * np.exp(r * t[k2]) - b
        error2 = (predicted2 - co2[k2]) ** 2
        total2 += error2
    result2 = np.sqrt(total2 / n)
    return result2


exponential_function_wrapper = lambda coeffs: exponential_function(coeffs[0], coeffs[1], coeffs[2])
A3_temp = scipy.optimize.minimize(exponential_function_wrapper, np.array([30, 0.03, 300]), method="Nelder-Mead").x
A3 = np.reshape(A3_temp, (1, 3))
A4 = exponential_function_wrapper(A3_temp)

# RMS_Error = lambda coeffs: np.sqrt((1 / n) * np.sum(((coeffs[0] * np.exp(coeffs[1] * t) + coeffs[2]) - co2) ** 2))
# coeff_min = scipy.optimize.minimize(RMS_Error, np.array([30, 0.03, 300]), method='Nelder-Mead')
# coeffs = coeff_min.x
# A3 = np.reshape(coeffs, (1, 3))
# A4 = (RMS_Error(coeffs))
# print(RMS_Error(coeffs))


def exponential_function_morearg(a, r, b, A, B, C):
    total2 = 0
    for k2 in range(n):
        predicted2 = a * np.exp(r * t[k2]) - b + A * np.sin(B * (t[k2] - C))
        error2 = (predicted2 - co2[k2]) ** 2
        total2 += error2
    result2 = np.sqrt(total2 / n)
    return result2


exponential_function_morearg_wrapper = lambda coeffs: exponential_function_morearg(coeffs[0], coeffs[1], coeffs[2],
                                                                                   coeffs[3], coeffs[4], coeffs[5])
A5_temp = scipy.optimize.minimize(exponential_function_morearg_wrapper, np.array([A3_temp[0], A3_temp[1], A3_temp[2],
                                                                                  -5, 4, 0]), method='Nelder-Mead',
                                  options={'maxiter': 10000}).x
A5 = np.reshape(A5_temp, (1, 6))
A6 = exponential_function_morearg_wrapper(A5_temp)

data = np.genfromtxt('lynx.csv', delimiter=',')
t = data[0, :]
pop = data[1, :]
n2 = t.size
pop[10] = 34
pop[28] = 27
interp_func = scipy.interpolate.interp1d(t, pop, kind='cubic')
A7 = interp_func(24.5)
A8_temp = np.polyfit(t, pop, 1)
A8 = np.reshape(A8_temp, (1, 2))

total3 = 0
for k3 in range(n2):
    predicted = A8_temp[0] * t[k3] + A8_temp[1]
    error = (predicted - pop[k3]) ** 2
    total3 += error
A9 = np.sqrt(total3 / n2)

A10_temp = np.polyfit(t, pop, 2)
A10 = np.reshape(A10_temp, (1, 3))
total4 = 0
for k3 in range(n2):
    predicted = A10_temp[0] * t[k3] ** 2 + A10_temp[1] * t[k3] + A10_temp[2]
    error = (predicted - pop[k3]) ** 2
    total4 += error
A11 = np.sqrt(total4 / n2)

A12_temp = np.polyfit(t, pop, 10)
A12 = np.reshape(A12_temp, (1, 11))
total5 = 0
for k3 in range(n2):
    predicted = A12_temp[0] * t[k3] ** 10 + A12_temp[1] * t[k3] ** 9 + A12_temp[2] * t[k3] ** 8 + \
                A12_temp[3] * t[k3] ** 7 + A12_temp[4] * t[k3] ** 6 + A12_temp[5] * t[k3] ** 5 + \
                A12_temp[6] * t[k3] ** 4 + A12_temp[7] * t[k3] ** 3 + A12_temp[8] * t[k3] ** 2 + \
                A12_temp[9] * t[k3] + A12_temp[10]
    error = (predicted - pop[k3]) ** 2
    total5 += error
A13 = np.sqrt(total5 / n2)


def lynx_function(coeffs, t_):
    m1 = coeffs[0]
    m2 = coeffs[1]
    m3 = coeffs[2]
    b1 = coeffs[3]
    b2 = coeffs[4]
    b3 = coeffs[5]
    t1 = coeffs[6]
    t2 = coeffs[7]
    if t_ <= t1:
        predicted2 = m1 * t_ + b1
    elif t_ <= t2:
        predicted2 = m2 * t_ + b2
    else:
        predicted2 = m3 * t_ + b3
    return predicted2


def lynx_RMS(coeffs):
    total2 = 0
    for k2 in range(n2):
        error2 = (lynx_function(coeffs, t[k2]) - pop[k2]) ** 2
        total2 += error2
    return np.sqrt(total2 / n2)


population_hunting_wrapper = lambda coeffs: lynx_RMS(coeffs)
A14_temp = scipy.optimize.minimize(population_hunting_wrapper, np.array([1, 0, -1, 40, 5, 30, 15.5, 20.5]),
                                   method='Nelder-Mead', options={'maxiter': 10000}).x
A14 = A14_temp[6: 8].reshape(1, 2)
A15 = population_hunting_wrapper(A14_temp)
