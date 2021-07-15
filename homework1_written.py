# import numpy as np
# x = 3.1416
# y = 3.14159265358979644932
# z = np.pi
# print(x)
# print(y)
# print(z)
# print(y - z)
# print(x - z)

import numpy as np
import math
import matplotlib.pyplot as plt
x = np.arange(-5, 5, 0.5)
y1 = x
y2 = x - x ** 3/math.factorial(3)
y3 = x - x ** 3/math.factorial(3) + x ** 5/math.factorial(5)
y4 = x - x ** 3/math.factorial(3) + x ** 5/math.factorial(5) - x ** 7/math.factorial(7)

plt.plot(x, np.sin(x), "k")

plt.plot(x, y1, "y")

plt.plot(x, y2, "r")

plt.plot(x, y3, "b")

plt.plot(x, y4, "g")
plt.legend(("y = sin(x)", "T1 = x", "T2 = x-x^3/3!", "T3 = x-x^3/3!+x^5/5!", "T4 = x-x^3/3!+x^5/5!-x^7/7!"),
           loc="upper center")
plt.title("Taylor series approximation")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()
