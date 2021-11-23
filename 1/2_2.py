import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.01)
plt.plot(x, x - ((x ** 3)/6) + ((x**5/120)) -((x**7)/5040))
plt.show()
