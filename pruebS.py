import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
data = np.random.randn(2, 100)


plt.scatter(data[0], data[1])


plt.show()

