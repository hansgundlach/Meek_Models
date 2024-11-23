# %%

# read in ModelBenchDAtes.csv
import pandas as pd

# Read the CSV file
df = pd.read_csv("Datasets/Epoch_benchmarks.csv")
#remove nan entries in model size and dataset size
df = df.dropna(subset=["Model size"])
df = df.dropna(subset=["Dataset size"])
df = df.dropna(subset=["MMLU"])

# Sort data by the date to ensure it's in order
#extract the number of parameters and amount of data
# df['Parameters'] = df['Model size'].str.extract('(\d+)')
# df['Data'] = df['Dataset size'].str.extract('(\d+)')
df['Parameters'] = pd.to_numeric(df['Model size'].astype(str).apply(lambda x: float(x)))
df['Data'] = pd.to_numeric(df['Dataset size'].astype(str).apply(lambda x: float(x)))
#%%
print(df['Parameters'])



#%%

L0 = 1.69
A = 406.4
B = 410.7
alpha = 0.34
beta = 0.28
#compute the loss based on parameters and data size
loss = lambda x, y: (A/x**alpha)+(B/y**beta) +L0
df['Loss'] = loss(df['Parameters'], df['Data'])

# %%
#make a scatter of loss vs MMLU
import matplotlib.pyplot as plt
# plt.scatter(df['Loss'], df['MMLU'])
# plt.xlabel('Loss')
# plt.ylabel('MMLU')
# plt.title('Loss vs MMLU')
# plt.grid()
# plt.show()

# reverse the x axis so we can see the sigmoid
# plt.scatter(df['Loss'], df['MMLU'])
# plt.xlabel('Loss')
# plt.ylabel('MMLU')
# plt.title('Loss vs MMLU')
# plt.grid()
# plt.gca().invert_xaxis()
# #fit a sigmoid to the data to reverse the x axis
# from scipy.optimize import curve_fit
# import numpy as np
# def sigmoid(x, L ,x0, k, b):
#     y = L / (1 + np.exp(-k*(x-x0)))+b
#     return (y)
# p0 = [max(df['MMLU']), np.median(df['Loss']),1,min(df['MMLU'])] # this is an mandatory initial guess
# popt, pcov = curve_fit(sigmoid, df['Loss'], df['MMLU'], p0, method='dogbox', maxfev=10000)
# print(popt)
# plt.plot(df['Loss'], sigmoid(df['Loss'], *popt), 'r', label='fit')
# plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Assuming you have your DataFrame 'df' ready
# df = pd.read_csv('your_data.csv')  # Replace with your data source

# Example data (remove this and use your actual DataFrame)
# For demonstration purposes, here's some synthetic data resembling a sigmoid
# x = np.linspace(0, 10, 100)
# y = 100 / (1 + np.exp(-1 * (x - 5))) + np.random.normal(0, 5, x.size)
# df = pd.DataFrame({'Loss': x, 'MMLU': y})

x_data = df['Loss'].values
y_data = df['MMLU'].values

def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

initial_guess = [
    max(y_data),          # L
    np.median(x_data),    # x0
    1,                    # k
    min(y_data)           # b
]

try:
    popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=initial_guess, maxfev=10000)
    L, x0, k, b = popt
    print(f"Fitted parameters:\nL = {L}\nx0 = {x0}\nk = {k}\nb = {b}")
except RuntimeError as e:
    print("Error - curve_fit failed:", e)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Data', color='blue')

x_fit = np.linspace(min(x_data), max(x_data), 500)
y_fit = sigmoid(x_fit, *popt)

plt.plot(x_fit, y_fit, color='red', label='Fitted Sigmoid')

plt.xlabel('Loss')
plt.ylabel('MMLU')
plt.title('Loss vs MMLU with Fitted Sigmoid')
plt.grid(True)
plt.gca().invert_xaxis()
plt.legend()
plt.savefig("Figures/Loss_vs_MMLU.png")
plt.show()

# %%
\frac{L}{1 + e^{-k(x - x_0)}} + b