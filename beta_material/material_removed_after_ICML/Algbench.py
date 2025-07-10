#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.dates import date2num, DateFormatter
from sklearn.linear_model import LinearRegression
import numpy as np

#%%

# df contains data on from Futuretech on algorthmic improvements in language models
df = pd.read_csv("Datasets/Alg_bench.csv", parse_dates=["Publication date"])
df["date"] = pd.to_datetime(df["Publication date"])
# convert params to numeric
df["Parameters"] = pd.to_numeric(df["Parameters"], errors='coerce')

#%%

#select elements with goood perplexity
df.dropna(subset=["Perplexity (WT103)"])
df.dropna(subset=["Parameters"])
df.dropna(subset=["date"])
good = df
# conver to numeric
good["Perplexity (WT103)"] = pd.to_numeric(good["Perplexity (WT103)"], errors='coerce') 


#%%
good = good[good["Perplexity (WT103)"] < 40]


#%%
#graph minimum parameters to get good MMLU over time 
good.sort_values("date", inplace=True)
maxs = []
firstmindate = []
firstminparams = []
minparam = float("inf")
print(minparam, "minparam")

#%%
for i in range(len(good)):
    # maxs.append(good["Parameters"][:i].min())
    if minparam > good["Parameters"][i]:
        print(good["Parameters"][i], "good")
        firstmindate.append(good["date"][i])
        firstminparams.append(good["Parameters"][i])
        minparam = good["Parameters"][i]
# plt.scatter(good["date"], maxs, color="blue")
plt.scatter(firstmindate, firstminparams, color="red")
plt.yscale('log')


#%% 
# only graph the date an parameters that first reach the minimum perplexity

# %%


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your dataframe

# Drop rows with NaN values in specific columns
df.dropna(subset=["Perplexity (WT103)", "Parameters", "date"], inplace=True)

# Convert "Perplexity (WT103)" to numeric
df["Perplexity (WT103)"] = pd.to_numeric(df["Perplexity (WT103)"], errors='coerce')

# Filter rows with "Perplexity (WT103)" < 40
good = df[(df["Perplexity (WT103)"]<20) & (df["Perplexity (WT103)"] > 18)]

# Sort the dataframe by date
good.sort_values("date", inplace=True)
#%%
plt.scatter(good["date"], good["Parameters"], color="blue")
plt.yscale('log')


#%%

# Initialize lists to store the first minimum parameters and corresponding dates
firstmindate = []
firstminparams = []
minparam = float("inf")

# Loop through the dataframe to find the first minimum parameters
for i in range(len(good)):
    if minparam > good["Parameters"].iloc[i]:
        firstmindate.append(good["date"].iloc[i])
        firstminparams.append(good["Parameters"].iloc[i])
        minparam = good["Parameters"].iloc[i]

# Plotting the results
plt.scatter(firstmindate, firstminparams, color="red")
# plt.yscale('log')
plt.xlabel("Date")
plt.ylabel("Parameters")
plt.title("First Minimum Parameters Over Time")
plt.show()

# %%

