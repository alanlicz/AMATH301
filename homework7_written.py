import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

P = lambda x: 1 / np.sqrt(2 * np.pi * 0.55 ** 2 ) * np.exp((-(x -3.39) ** 2) / (2 * 0.55 ** 2))
Int, err = scipy.integrate.quad(P, 2, 4)

dx_array = np.zeros(15)
LHR = np.zeros(15)
RHR = np.zeros(15)
trap = np.zeros(15)
for i in range(15):
    dx = 2**-(i+2)
    dx_array[i] = dx
    x = np.arange(2, 4 + dx, dx)
    y = P(x)
    LHR[i] = dx * np.sum(y[:-1])
    RHR[i] = dx * np.sum(y[1:])
    trap[i] = (LHR[i] + RHR[i]) / 2
LHR = np.abs(LHR - Int)
RHR = np.abs(RHR - Int)
trap = np.abs(trap - Int)
plt.xlabel("dx")
plt.ylabel("error")
plt.title("Errors vs dx for each method")
plt.plot(dx_array, LHR, 'ro', dx_array, RHR, 'yo', dx_array, trap, 'bo')
plt.legend(("LHR error", "RHR error", "Trapezoidal error"), loc="best")
plt.show()

logx = np.log(dx_array)
logL = np.log(LHR)
logR = np.log(RHR)
logT = np.log(trap)
plt.xlabel("log(dx)")
plt.ylabel("log(error)")
plt.title("Log error vs log dx for each method")
plt.plot(logx, logL, 'ro', logx, logR, 'yo', logx, logT, 'bo')
plt.legend(("LHR log error", "RHR log error", "Trapezoidal log error"), loc="best")
plt.show()

coeffsL = np.polyfit(logx, logL, 1)
coeffsR = np.polyfit(logx, logR, 1)
coeffsT = np.polyfit(logx, logT, 1)
xplot = np.linspace(logx[0], logx[-1], 1000)
yplotL = np.polyval(coeffsL, xplot)
yplotR = np.polyval(coeffsR, xplot)
yplotT = np.polyval(coeffsT, xplot)
plt.xlabel("log(dx)")
plt.ylabel("log(error)")
plt.title("Log error vs log dx for each method")
plt.plot(logx, logL, 'ro', logx, logR, 'yo', logx, logT, 'bo')
plt.plot(xplot, yplotL, 'r', xplot, yplotR, 'y', xplot, yplotT, 'b')
plt.legend(("LHR log error", "RHR log error", "Trapezoidal log error", "LHR log error line", "RHR log error line",
            "Trapezoidal log error line"), loc="best")
plt.show()

print("slope for LHR log error line is ", coeffsL[0])
print("slope for RHR log error line is ", coeffsR[0])
print("slope for Trapezoidal log error line is ", coeffsT[0])