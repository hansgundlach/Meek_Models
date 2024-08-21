

#%%
# #plot training and inference overhang 
# plt.figure(figsize=(10, 5))
# plt.plot(time, player1loss/total_loss(1000, time), label="Training Overhang")



# %%
# look at perplexity/model size over time

# plt.figure(figsize=(10, 5))  # Set the size of the plot
# plt.scatter(
#     alg["date_num"], alg["Perplexity (PTB)"]/np.log(alg["Parameters"]), marker="o"
# )  # Line plot with markers
# # plt.yscale("log")


# %%
# graph of memory with proper sizd


# Assuming 'hard' and 'df' are your dataframes with the 'date_num' columns containing days since the Unix epoch

# Convert days since epoch to datetime objects
hard_dates = [
    datetime.datetime(1970, 1, 1) + datetime.timedelta(days=int(ts))
    for ts in hard["date_num"]
]
df_dates = [
    datetime.datetime(1970, 1, 1) + datetime.timedelta(days=int(ts))
    for ts in df["date_num"]
]

# Memory size plot
plt.figure(figsize=(10, 5))  # Set the size of the plot
plt.scatter(
    hard_dates, hard["Memory size per board (Byte)"], marker="o"
)  # Scatter plot for 'hard' data
plt.scatter(
    df_dates, df["Parameters"], marker="o", color="orange"
)  # Scatter plot for 'df' data

# Format the x-axis to show years with labels every five years
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=5))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Rotate and align the tick labels so they look better
plt.gcf().autofmt_xdate()

# Add grid, legend, and labels
plt.grid(True)  # Add grid for better readability
plt.legend(["Memory size per board (Byte)", "Parameters"])
plt.yscale("log")  # Set y-axis to logarithmic scale
plt.xlabel("Year")
plt.ylabel("Values")
plt.title("Memory Size and Parameters Over Time")

# Show the plot
plt.show()


# junk code I'm not using currently

# same plot now with matplotlib
import plotly.graph_objs as go
import pandas as pd
import datetime

# Assuming 'hard' and 'df' are your dataframes with the 'date_num' columns containing days since the Unix epoch

# Convert days since epoch to datetime objects
hard_dates = [
    datetime.datetime(1970, 1, 1) + datetime.timedelta(days=int(ts))
    for ts in hard["date_num"]
]
df_dates = [
    datetime.datetime(1970, 1, 1) + datetime.timedelta(days=int(ts))
    for ts in df["date_num"]
]

# Create traces for the scatter plot
trace1 = go.Scatter(
    x=hard_dates,
    y=hard["Memory size per board (Byte)"],
    mode="markers",
    name="Memory size per board (Byte)",
    marker=dict(color="blue"),
)

trace2 = go.Scatter(
    x=df_dates,
    y=df["Parameters"],
    mode="markers",
    name="Parameters",
    marker=dict(color="orange"),
)

# Create the figure
fig = go.Figure(data=[trace1, trace2])

# Update layout with appropriate axis settings
fig.update_layout(
    title="Memory Size and Parameters Over Time",
    xaxis=dict(
        title="Year",
        type="date",
        tickformat="%Y",
        dtick="M60",  # Tick every five years
    ),
    yaxis=dict(
        title="Values",
        type="log",
    ),
    showlegend=True,
    width=1000,
    height=500,
)

# Add grid settings
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

# Show the plot
fig.show()

# plt.plot(time, player1loss-player2loss, label="Second Best Model")
# plt.plot(
#     time,
#     np.log(logistic_loss/total_loss(1000, time)),
#     label="Loss Diff logisitic investment vs 1000 dollar training run",
# )

# plt.plot(
#     time,
#     player1loss-player2loss,
#     label="Exponential Investment over staggered exponential growth",
# )

# plt.xlabel("Time (years)")
# plt.title("AI Model Inequality Over Time")
# plt.xlabel("Time (years)")
# plt.legend()
# plt.ylabel("Log Likelihood Loss Difference")
# plt.grid(True)




# # Symbolic computations for overhang
# # Define the variables
# cost, time = sp.symbols("cost time")
# g_alg, C0, I, t, g_flop, g_invest = sp.symbols("g_alg C0 I t g_flop g_invest")
# L0, A, b, K, g = sp.symbols("L0, A, b, K, g")
# # Define the function
# chin_func = A * I ** (b) + L0
# # total_loss = chin_func.subs(I, (g_alg**t) * C0 * K * g_flop**t)
# best_model_form = chin_func.subs(I, (g**t) * g_invest**t)
# standard_model_form = chin_func.subs(I,g**t )
# # Simplify and display the function
# # total_loss = sp.simplify(total_loss)
# overhang = best_model_form-standard_model_form
# # overhang = sp.simplify(overhang)

# print(latex(total_loss))
# sp.pprint(total_loss)
# print("The overhang is")
# sp.pprint(overhang)
#%%
# # calculate when the derivative of overhang is zero wrt time
# # take derivative of overhang wrt time  
# # overhang = overhang.subs(C0, 1000)
# # overhang = overhang.subs(g_alg, 1.8)
# # overhang = overhang.subs(g_flop, 0.8)
# # overhang = overhang.subs(g_invest, 5)
# # overhang = overhang.subs(I, 1)
# # overhang = sp.simplify(overhang)
# print("The overhang is")
# sp.pprint(overhang)
# overhang_derivative = sp.diff(overhang, t)
# overhang_derivative = sp.simplify(overhang_derivative)
# print("The derivative of the overhang is")
# sp.pprint(overhang_derivative)
# # get numerator and denominator of derivative
# numerator, denominator = sp.fraction(overhang_derivative)
# numerator = sp.simplify(numerator)
# # find the roots of the numerator
# roots = sp.solve(numerator, t)

# # overhang = sp.simplify(overhang)
# # print("The derivative of the overhang is")
# # sp.pprint(overhang)

##%
# all the stuff for logisitcs 
# time = np.linspace(0, 2, 100)
# plt.figure(figsize=(10, 5))
# # logistic_investment = lambda time: 1000 + 1e10/ (1 + 1e5*np.exp(-np.log(5)* (time+1)))
# plt.plot(time, logistic_investment(time), label="Logistic Investment")
# plt.plot(time, 1000 * (player1_growth) ** time, label="Exponential Investment")
# plt.xlabel("Time (years)")
# plt.ylabel("Investment")
# plt.legend()


#%%%
# logistic_investment = lambda time: 1000 + 10000/ (1 + np.exp(-1 * (time)))
# logistic_investment = lambda time: 1000 + 5e9/ (1 + (2e9)*np.exp(-5* (time+2)))
# logistic_loss = total_loss(logistic_investment(time), time)

#make investment schedule where invesemtent increasese exponentially but then stops 
#
# add description to plot



# plt.plot(time, player1loss-player2loss, label="Second Best Model")
# plt.plot(
#     time,
#     np.log(logistic_loss/total_loss(1000, time)),
#     label="Loss Diff logisitic investment vs 1000 dollar training run",
# )

# plt.plot(
#     time,
#     player1loss-player2loss,
#     label="Exponential Investment over staggered exponential growth",
# )

# plt.xlabel("Time (years)")


#==============================================================================
# %%
# model of training overhang over time
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import latex

# %%
cost = 1000
cost_per_flop_year = 0.8
flops_per_dollar_year0 = 1e9 / 0.02
alg_gains_train = 1.8
alg_gains = alg_gains_train

# Chinchilla function relating compute to perplexity
chin_func = lambda x: 1070 * x ** (-0.154) + 1.7
total_loss = lambda cost, time: chin_func(
    (alg_gains**time) * flops_per_dollar_year0 * cost / (cost_per_flop_year**time)
)

# %%
# Graph this function with respect to time
time = np.linspace(0, 100, 100)
player1_growth = 1.5
player2_growth = 1.1

# Loss for best model over time
player2loss = total_loss(1000 * (player2_growth) ** time, time)
player1loss = total_loss(1000 * (player1_growth) ** time, time)
logistic_investment = lambda time: 10000 / (1 + np.exp(-0.1 * (time - 10)))
logistic_loss = total_loss(logistic_investment(time), time)

# Graph with multiple costs adding cost labels
plt.figure(figsize=(10, 5))
for c in [1000, 10000, 100000]:
    loss = total_loss(c, time)
    plt.plot(time, player1loss, label="Exponential Investment")
    plt.plot(time, loss, label=f"Cost: {c}")
    plt.plot(time, logistic_loss, label="Logistic Investment")

plt.xlabel("Time (years)")
plt.ylabel("Total Loss")
plt.legend()
plt.title("Training Loss Over Time")
plt.grid(True)

# %%
# Plot of overhang over time
plt.figure(figsize=(10, 5))
plt.plot(
    time, player1loss / total_loss(1000, time), label="Overhang over constant player"
)
plt.plot(time, player1loss / player2loss, label="Second Best Model")
plt.plot(
    time,
    logistic_loss / total_loss(1000, time),
    label="Overhang over logistic investment",
)

plt.xlabel("Time (years)")
plt.ylabel("Perplexity Overhang")
plt.legend()
plt.title("Training Overhang Over Time")
plt.grid(True)

# %%
# Model of inference overhang with a set number of parameters
player1_growth = 1.9
player2_growth = 1.0
alg_gains_inf = 1.2

total_loss_inf = lambda cost, time: chin_func(
    (
        (cost / (cost_per_flop_year**time)) ** 2
        * (alg_gains_train**time)
        * flops_per_dollar_year0
    )
)

time = np.linspace(0, 40, 10000)
plt.figure(figsize=(10, 5))
for c in [1000, 10000, 100000]:
    loss = total_loss(c, time)
    plt.plot(time, loss, label=f"Cost: {c}")

best_loss_seq = total_loss(1 * (player1_growth) ** time, time)
inf_overhang = total_loss_inf(100000, time) / total_loss_inf(100, time)
inf_bestvs1000dollars = best_loss_seq / total_loss_inf(1, time)

plt.ylabel("Total Loss for Models")
plt.xlabel("Time (years)")
plt.title("Inference Overhang for Given Inference Cost")
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 5))
plt.plot(
    time,
    inf_overhang,
    "r",
    label="Inference Overhang 100000 dollar model vs 100 dollar model",
)
plt.plot(
    time,
    inf_bestvs1000dollars,
    label="Inference Overhang Best Model vs 1000 dollar model",
)

plt.title("Inference Overhang of 10000 dollar model vs 100 dollar model Over Time")
plt.xlabel("Time (years)")
plt.ylabel("Overhang of Inference")
plt.legend()
plt.grid(True)

# %%
# Plot inference overhang and training overhang at the same time
player1loss = total_loss(1000 * (player1_growth) ** time, time)
plt.figure(figsize=(10, 5))
plt.plot(time, player1loss/total_loss_inf(1, time), label="Inference Overhang")
plt.plot(time, player1loss / total_loss(1000, time), label="Training Overhang")

plt.xlabel("Time (years)")
plt.ylabel("Overhang")
plt.legend()
plt.title("Inference and Training Overhang Over Time")
plt.grid(True)

# %%
# Inference overhang for constant parameter models
time = np.linspace(0, 100, 10000)
cost_per_flop_year = 1
loss_constant = 1.7
const_params = 1
A = 1
loss_const_param = (
    lambda time: loss_constant + A / (const_params * alg_gains**time) ** 0.34
)

plt.figure(figsize=(10, 5))
plt.xlabel("Time (years)")
plt.ylabel("Inference Overhang")
world_loss_seq = total_loss(1000 * (player1_growth) ** time, time)
loss_infoptimal_year_seq = loss_const_param(time)
plt.plot(time, world_loss_seq / loss_infoptimal_year_seq, label="Inference Overhang")

plt.title("world_best/const_param Over Time")
plt.legend()
plt.grid(True)

# %%
# Another model of inference overhang assuming fixed parameters for better and base model
time = np.linspace(0, 40, 10000)
plt.figure(figsize=(10, 5))
plt.plot(time, 1.5 ** (-0.034 * time), label="Inference Overhang")

# %%
# Symbolic computations for overhang
# Define the variables
cost, time = sp.symbols("cost time")
g_alg, C0, I, t, g_flop, g_invest = sp.symbols("g_alg C0 I t g_flop g_invest")

# Define the function
chin_func = 1070 * I ** (-0.154) + 1.7
total_loss = chin_func.subs(I, (g_alg**t) * C0 * I * g_flop**t)

# Simplify and display the function
total_loss = sp.simplify(total_loss)
overhang = total_loss.subs(
    I, (g_alg**t) * g_invest**t * C0 * I * g_flop**t
) / total_loss.subs(I, C0 * I * (g_alg**t) * g_flop**t)
overhang = sp.simplify(overhang)

print(latex(total_loss))
sp.pprint(total_loss)
print("The overhang is")
sp.pprint(overhang)

# %%
