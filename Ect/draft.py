#%%
#graph the difference of two sigmoids functions
import numpy as np
import matplotlib.pyplot as plt


#%%
def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.linspace(0, 10, 100)
y1 = sigmoid(x)**10
y2 = sigmoid(np.log(x))**10
y3 = y1-y2

# plt.plot(x, y1, label='sigmoid(x)')
plt.plot(x, y2, label='sigmoid(2*x)')
# plt.plot(x, y3, label='sigmoid(x)-sigmoid(2*x)')
plt.legend()
plt.xscale('log')
# %%
