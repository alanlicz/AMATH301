import numpy as np
import matplotlib.pyplot as plt
# import scipy.optimize
# import scipy.interpolate
#
# data = np.genfromtxt('CO2_data.csv', delimiter=',')
# t = data[0, :]
# co2 = data[1, :]
# n = t.size
#
# plt.plot(t, co2, "k")
# plt.title("Atmospheric CO2 at Mauna Loa")
# plt.xlabel("time(in years since January 1958)")
# plt.ylabel("CO2(in ppm)")
#
# coeffs = np.polyfit(t, co2, 1)
# yplot2 = coeffs[0] * t + coeffs[1]
# plt.plot(t, yplot2, "b")
#
# RMS_Error = lambda coeffs_: np.sqrt((1 / n) * np.sum(((coeffs_[0] * np.exp(coeffs_[1] * t) + coeffs_[2]) - co2) ** 2))
# coeffs2 = scipy.optimize.minimize(RMS_Error, np.array([30, 0.03, 300]), method='Nelder-Mead').x
# yplot3 = coeffs2[0] * np.exp(coeffs2[1] * t) + coeffs2[2]
# plt.plot(t, yplot3, "r")
#
# RMS_Error = lambda coeffs_: np.sqrt((1 / n) * np.sum(((coeffs_[0] * np.exp(coeffs_[1] * t) + coeffs_[2] + coeffs_[3] *
#                                                        np.sin(coeffs_[4] * (t - coeffs_[5]))) - co2) ** 2))
# coeffs3 = scipy.optimize.minimize(RMS_Error, np.array([coeffs2[0], coeffs2[1], coeffs2[2], -5, 4, 0]),
#                                   method='Nelder-Mead', options={'maxiter': 10000}).x
# yplot4 = coeffs3[0] * np.exp(coeffs3[1] * t) + coeffs3[2] + coeffs3[3] * np.sin(coeffs3[4] * (t - coeffs3[5]))
# plt.plot(t, yplot4, "c")
#
# plt.legend(("actual data point", "f(t) = mt + b", "f(t) =ae^rt + b", "f(t) = ae^rt + b + A sin(B(t − C))"),
#            loc="best")
#
# plt.show()
# print(coeffs[0] * (62 + 11 / 12) + coeffs[1])
# print(coeffs2[0] * np.exp(coeffs2[1] * (62 + 11 / 12)) + coeffs2[2])
# print(coeffs3[0] * np.exp(coeffs3[1] * (62 + 11 / 12)) + coeffs3[2] + coeffs3[3] *
#       np.sin(coeffs3[4] * ((62 + 11 / 12) - coeffs3[5])))
# interp_func = scipy.interpolate.interp1d(t, co2, kind='cubic', fill_value="extrapolate")
# print(interp_func(62 + 11 / 12))
# interp_func2 = scipy.interpolate.interp1d(t, co2, fill_value="extrapolate")
# print(interp_func2(62 + 11 / 12))
t = np.array(np.linspace(2015, 2020, 6))
y = np.random.rand(t.size)
plt.figure()
plt.plot(t, y, 'ko')
plt.xlabel("t")
plt.ylabel("y")
plt.title("y vs t")
plt.show()

coeffs = np.polyfit(t, y, 5)
print("The coefficients are: ", coeffs)
x_at_2020 = np.polyval(coeffs, 2020)
print("The value of p(2020) is ", x_at_2020)
print("The error is: ", abs(x_at_2020 - y[-1]))

print(coeffs[-1] * (2020 ** 5))
print(coeffs[-2] * (2020 ** 4))
print(coeffs[-3] * (2020 ** 3))
print(coeffs[-4] * (2020 ** 2))
print(coeffs[-5] * (2020 ** 1))
print(coeffs[-6])
