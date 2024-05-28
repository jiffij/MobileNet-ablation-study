import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

alpha = 0.2
x = np.linspace(0, 1, 100)
y = beta.pdf(x, alpha, alpha)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Beta Distribution")
plt.show()

