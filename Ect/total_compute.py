# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.dates import date2num, DateFormatter
from sklearn.linear_model import LinearRegression
import numpy as np

# %%
# df contains epoch data on large models
df = pd.read_csv("notable_systems.csv")
# hard contains hardware data over time
hard = pd.read_csv("ML_hardware.csv")
# alg contains data from study of algorithmic improvmenents in language models
alg = pd.read_csv("Alg_bench.csv")

# %%
hard_names = hard.columns.tolist()
print(hard_names)

# format and read hardware datbase
hard["date"] = pd.to_datetime(hard["Release year"].astype(str) + "-01-01")
hard["date_num"] = hard["date"].apply(date2num)

# need computer per dollar column

df = df.dropna(subset=["Inference compute (FLOP)"])
# Sort data by the date to ensure it's in order
df.sort_values("Publication date", inplace=True)

df["date"] = pd.to_datetime(df["Publication date"])
df["date_num"] = df["date"].apply(date2num)

# %%

# find all rows that have parameter count less then 1 billion and greater than 1 miilion

alg = alg.dropna(subset=["Parameters"])

# convert to numeric
alg["Parameters"] = pd.to_numeric(alg["Parameters"], errors="coerce")
alg["date"] = pd.to_datetime(alg["Publication date"])
alg["date_num"] = alg["date"].apply(date2num)



condition = (alg["Parameters"] < 1e8) & (alg["Parameters"] > 1e7)
algsmall = alg[condition]
algsmall = algsmall.dropna(subset=["Perplexity (WT103)"])
# algsmalltest = alg[(alg["Parameters"] < 1e6) & (alg["Parameters"] >  1e6)]
# algsmalltest = alg.dropna(subset=["Perplexity (PTB)"])


bigcondition = (alg["Parameters"] > 1e8) & (alg["Parameters"] < 1e9)

# find all rows that have parameters count greater than 1 billion
algbig = alg[bigcondition]
algbig = algbig.dropna(subset=["Perplexity (WT103)"])
# %%

# print rows of algsmall
print(algsmall)
# print size of algsmall
print(algsmall.shape)
print(alg.shape, "full shape")

# plot perplexity vs time for algsmall
# algsmall["date"] = pd.to_datetime(algsmall["Publication date"])
# algsmall["date_num"] = algsmall["date"].apply(date2num)

# algbig["date"] = pd.to_datetime(algbig["Publication date"])
# algbig["date_num"] = algbig["date"].apply(date2num)


# create fit for algsmall:
X = algsmall["date_num"].values.reshape(-1, 1)
y = algsmall["Perplexity (WT103)"].values
reg = LinearRegression().fit(X, y)
y_pred_small = reg.predict(X)

slope = reg.coef_[0]
r_squared = reg.score(X, y)
print(f"Slope: {slope}, R^2: {r_squared}")


# %%
# create fit for algbig
X = algbig["date_num"].values.reshape(-1, 1)
y = algbig["Perplexity (WT103)"].values
reg = LinearRegression().fit(X, y)
y_pred_big = reg.predict(X)

slope = reg.coef_[0]
r_squared = reg.score(X, y)
print(f"Slope for Large Models: {slope}, R^2: {r_squared}")


# %%
plt.figure(figsize=(10, 5))  # Set the size of the plot
plt.scatter(
    algsmall["date_num"], algsmall["Perplexity (WT103)"], marker="o"
)  # Line plot with markers
plt.scatter(algbig["date_num"], algbig["Perplexity (WT103)"], marker="x")
# plt.yscale("log")
# plt.plot(algsmall["date_num"], y_pred_small, color="b", label="Linear Fit (PTB)")
# plt.plot(algbig["date_num"], y_pred_big, color="r", label="Linear Fit (WT2)")
plt.ylabel("Perplexity")
plt.xlabel("Publication Date")
plt.title("Perplexity Over Time For Large and Big Models")
plt.yscale("log")





# %%


alg = alg.dropna(subset=["Perplexity (PTB)"])
# Filter the data for performance > 50 and find the minimum size for each year
filtered_df = alg[(alg['Perplexity (PTB)'] < 70) & (alg['Perplexity (PTB)'] < 50)]


min_size_per_year = filtered_df.groupby('Year')['Parameters'].mean().reset_index()

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(min_size_per_year['Year'], min_size_per_year['Parameters'], marker='o', linestyle='-')
plt.title('Minimum Size to Achieve Performance < 50 Over Time')
plt.xlabel('Year')
plt.ylabel('Minimum Size')
plt.yscale('log')
plt.grid(True)
plt.show()





#%%


# tracking price of compute for GPU's over time
print(hard.columns.tolist())
pricedata = hard.dropna(
    subset=[
        "Cloud pricing ($ per hour) data from 03 July 2023; Google cloud and lambda labs prices"
    ]
)


pricedata["Price"] = pricedata[
    "Cloud pricing ($ per hour) data from 03 July 2023; Google cloud and lambda labs prices"
].replace({"\$": "", ",": ""}, regex=True)

# Step 3: Convert to numeric
pricedata["Price"] = pd.to_numeric(pricedata["Price"])
pricedata["Price/s"] = pricedata["Price"] / 3600

pricedata["date"] = pd.to_datetime(pricedata["Release year"].astype(str) + "-01-01")
plt.scatter(pricedata["date"], pricedata["Price"], marker="o")
plt.ylabel("Price per hour cloud")
plt.xlabel("Hardware Release Date")


# %%
# get price performance data
pricedata = pricedata.dropna(subset=["FP32 Performance (FLOP/s)"])
pricedata["Price Performance"] = (
    pricedata["FP32 Performance (FLOP/s)"] / pricedata["Price/s"]
)
plt.scatter(pricedata["date"], pricedata["Price Performance"], marker="o")
plt.ylabel("Price Performance")
plt.xlabel("Hardware Release Date")
plt.grid(True)
plt.yscale("log")
# %%
# make graphs of each time of overhang measure
# this includes dollars needed for given level of performance
# this also includes total ration of total compute/lowest inference cost for given performance


# world total compute esimates
total = pd.read_csv("WorldTotalCompute.csv")
print(total.head())
plt.figure(figsize=(10, 5))  # Set the size of the plot
plt.scatter(total["year"], total["size"], marker="o")  # Line plot with markers
plt.grid(True)
plt.title("Spending on GPU Compute Over Time")  # Title of the plot


# %%
# %%
# Plotting the data
plt.figure(figsize=(10, 5))  # Set the size of the plot
plt.scatter(
    df["date_num"], df["Inference compute (FLOP)"], marker="o"
)  # Line plot with markers
plt.scatter(df["date_num"], df["Training compute (FLOP)"], marker="o")


plt.scatter(hard["date_num"], hard["FP32 Performance (FLOP/s)"], marker="o")
plt.scatter(pricedata["date"], pricedata["Price Performance"], marker="o")

plt.title("Inference Compute Over Time")  # Title of the plot
plt.xlabel("Publication Date")  # Label for the x-axis
plt.ylabel("Inference Compute")  # Label for the y-axis
plt.yscale("log")
plt.grid(True)  # Add grid for better readability
plt.xticks(rotation=45)  # Rotate date labels for better visibility
plt.legend(
    [
        "Inference compute (FLOP)",
        "Training compute (FLOP)",
        "FP32 Performance (FLOP/s)",
        "Price Performance",
    ]
)

plt.tight_layout()  # Adjust layout to not cut off labels


# make linear fit fro price performance
X = pricedata["date_num"].values.reshape(-1, 1)
y = np.log(pricedata["Price Performance"].values)
reg = LinearRegression().fit(X, y)

y_pred_price = np.exp(reg.predict(X))

# graph the fit
plt.plot(
    pricedata["date_num"],
    y_pred_price,
    color="r",
    label="Linear Fit (Price Performance)",
)

# Predict price performance for the years in the total compute dataset
predictpricep = np.exp(reg.predict(total["year"].values.reshape(-1, 1)))

# Divide total compute by predicted price performance
total["World Flop/s"] = total["size"] * predictpricep


# convert year to datetiem


total["date"] = pd.to_datetime(total["year"].astype(str) + "-01-01")
plt.scatter(total["date"], total["World Flop/s"], marker="o", color="purple")

plt.show()
# %%

# memory size plot
plt.figure(figsize=(10, 5))  # Set the size of the plot
plt.scatter(
    hard["date_num"], hard["Memory size per board (Byte)"], marker="o"
)  # Line plot with markers
plt.scatter(df["date_num"], df["Parameters"], marker="o", color="orange")
plt.grid(True)  # Add grid for better readability
plt.legend(["Memory size per board (Byte)", "Parameters"])
plt.yscale("log")


# %%


# graph performance/model size by year 


# %%
