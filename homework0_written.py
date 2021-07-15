# import numpy as np
# import matplotlib.pyplot as plt
# x = np.arange(-2, 2.25, 0.25)
# y = 2*x + 3
# plt.figure()
# plt.plot(x, y, 'k')
# plt.plot(x, y, 'bo')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Plot of a line')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-2, 2.25, 0.25)
y = x**2 - 1
plt.figure()
plt.plot(x, y, 'k')
plt.plot(x, y, 'bo')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of a parabola')
plt.show()