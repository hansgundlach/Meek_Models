# %%


# model of training overhang over time

import numpy as np
import matplotlib.pyplot as plt

cost = 1000
cost_per_flop_year = 0.8
flops_per_dollar_year0 = 1e9 / 0.02
alg_gains = 1.8
#chinchilla function 
chin_func = lambda x: 1070 * x ** (-0.154) + 1.7
total_loss = lambda cost, time: chin_func(
    (alg_gains**time) * flops_per_dollar_year0 * cost / ((cost_per_flop_year) ** time)
)

# now graph this function with respect to time
time = np.linspace(0, 40, 100)

#%%
# graph with multiple costs adding cost labels
plt.figure(figsize=(10, 5))
for c in [1000, 10000, 100000]:
    loss = total_loss(c, time)

    plt.plot(time, loss, label=f"Cost: {c}")
    # plt.plot(time, loss, label=f"Cost: {c}")

player1_growth = 1.5
player2_growth = 1.7

#%%
# symbolic computations
import sympy as sp
from sympy import latex
# define the variables
cost, time = sp.symbols("cost time")
# define more varaibles
g_alg= sp.symbols("g_alg")
C0 = sp.symbols("C0") # flops per dollar year 0
I = sp.symbols("I") # investment
t = sp.symbols("t") # time
g_flop = sp.symbols("g_flop") # growth in flops per dollar per year
g_invest = sp.symbols("g_invest") # growth in investment per year 

# define the function
chin_func = 1070 * I ** (-0.154) + 1.7
# display function 
total_loss = chin_func.subs(I, (g_alg**t) * C0 * I * g_flop**t)
# display the function
total_loss
#simplify the function
total_loss = sp.simplify(total_loss)
#overhang amount  
overhang = total_loss.subs(I, (g_alg**t)*g_invest**t * C0 * I * g_flop**t) / total_loss.subs(I, C0 * I * (g_alg**t)*g_flop**t)

overhang = sp.simplify(overhang)




# display the simplified function
print(latex(total_loss))


# define the total loss function
# print(latex(chin_func))

#%%
sp.pprint(total_loss)
#the overhang is 
print("the overhang is")
sp.pprint(overhang)
#simplify exponential
total_loss = sp.simplify(total_loss)


#%%

# loss for best model over time
secondplayer = total_loss(1000 * (player1_growth) ** time, time)
toploss = total_loss(1000 * (player2_growth) ** time, time)
plt.plot(time, toploss, label="Best Model")
overhang = toploss / total_loss(1000, time)

plt.legend()
# loss = total_loss(cost, time)
# plt.plot(time, loss)
plt.xlabel("Time (years)")
plt.ylabel("Total Loss")

# %%
# new plot
plt.figure(figsize=(10, 5))
plt.plot(time, overhang, label="Overhang over constant player")
plt.plot(time, toploss / secondplayer, label="Second Best Model")
plt.xlabel("Time (years)")
plt.ylabel("Overhang")
plt.legend()
plt.title("Training Overhang Over Time")
# %%
player1_growth = 1.9
player2_growth = 1.0
alg_gains_inf = 1.2

total_loss_inf = lambda cost, time: chin_func(
    (
        (alg_gains**time)
        * flops_per_dollar_year0
        * cost
        / ((cost_per_flop_year) ** time)
    )
    ** 2
)
time = np.linspace(0, 40, 10000)
for c in [1000, 10000, 100000]:
    loss = total_loss(c, time)
    plt.plot(time, loss, label=f"Cost: {c}")
inf_overhang = total_loss_inf((alg_gains_inf**time)*100000, time) / total_loss_inf((alg_gains_inf**time)*100, time)
inf_overhang_variable = total_loss_inf((player1_growth) ** time, time) / total_loss_inf(
    (player2_growth) ** time, time
)
plt.ylabel("Total Loss")
plt.xlabel("Time (years)")
plt.legend()



plt.figure(figsize=(10, 5))
plt.plot(time, inf_overhang, label="Inference Overhang Constant player growth/constant")
plt.plot(time, inf_overhang_variable, label="Inference Overhang Variable Players growth1/growth2")
plt.title("Overhang of Best Model Over Time")
plt.xlabel("Time (years)")
plt.ylabel("Overhang of Inference")
plt.legend()

# inference overhang model


# this could model US china embargo relations
# this could also model two companies asking when to aquire


#%%
#modle of inference overhang with set number of parameters
time = np.linspace(0, 40, 10000)
cost_per_flop_year = 1
loss_chin_inf = lambda cost, time: chin_func((alg_gains**time) * flops_per_dollar_year0 * cost / ((cost_per_flop_year) ** time)
)
loss_constant = 1
const_params = 1
A = 1
loss_infoptimal_year = lambda time: loss_constant + A/(const_params*alg_gains**time)**0.34
plt.figure(figsize=(10, 5))
hang_best_constparam = loss_chin_inf(1000*5**time, time) / loss_infoptimal_year(time)
# set y axis scale from 0 to 5
plt.ylim(0, 5)
plt.plot(time, hang_best_constparam, label="Inference Overhang")
plt.title("world_best/inf_optima Over Time")









# %%
# another model of inference overhang
time = np.linspace(0, 40, 10000)
plt.figure(figsize=(10, 5))
plt.plot(time, 1.5 ** (-0.034 * time), label="Inference Overhang")
#%%


# import numpy as np
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# from ipywidgets import interact, FloatSlider

# def plot_loss(cost=1000, cost_per_flop_year=0.8, flops_per_dollar_year0=1e9 / 0.02, alg_gains=1.8, player1_growth=1.5, player2_growth=1.8):
#     # Chinchilla function
#     chin_func = lambda x: 1070 * x ** (-0.154) + 1.7
#     total_loss = lambda cost, time: chin_func(
#         (alg_gains**time) * flops_per_dollar_year0 * cost / ((cost_per_flop_year) ** time)
#     )

#     # Time range
#     time = np.linspace(0, 40, 10000)

#     # Create subplots
#     fig = make_subplots(rows=2, cols=1, subplot_titles=("Total Loss Over Time", "Training Overhang Over Time"))

#     # Plot total loss for multiple costs
#     for c in [1000, 10000, 100000]:
#         loss = total_loss(c, time)
#         fig.add_trace(go.Scatter(x=time, y=loss, mode='lines', name=f"Cost: {c}"), row=1, col=1)

#     # Loss for best model over time
#     secondplayer = total_loss(1000 * (player1_growth) ** time, time)
#     toploss = total_loss(1000 * (player2_growth) ** time, time)
#     fig.add_trace(go.Scatter(x=time, y=toploss, mode='lines', name="Best Model"), row=1, col=1)

#     overhang = toploss / total_loss(1000, time)

#     fig.add_trace(go.Scatter(x=time, y=overhang, mode='lines', name="Overhang over constant player"), row=2, col=1)
#     fig.add_trace(go.Scatter(x=time, y=toploss / secondplayer, mode='lines', name="Second Best Model"), row=2, col=1)

#     # Update layout
#     fig.update_layout(height=800, width=1000, title_text="Interactive Plots with Sliders")
#     fig.update_xaxes(title_text="Time (years)", row=1, col=1)
#     fig.update_yaxes(title_text="Total Loss", row=1, col=1)
#     fig.update_xaxes(title_text="Time (years)", row=2, col=1)
#     fig.update_yaxes(title_text="Overhang", row=2, col=1)

#     fig.show()

# # Interactive sliders for the variables
# interact(plot_loss, 
#          cost=FloatSlider(min=500, max=50000, step=500, value=1000),
#          cost_per_flop_year=FloatSlider(min=0.1, max=2, step=0.1, value=0.8),
#          flops_per_dollar_year0=FloatSlider(min=1e8, max=1e10, step=1e8, value=1e9 / 0.02),
#          alg_gains=FloatSlider(min=1, max=3, step=0.1, value=1.8),
#          player1_growth=FloatSlider(min=1, max=3, step=0.1, value=1.5),
#          player2_growth=FloatSlider(min=1, max=3, step=0.1, value=1.8))

# %%
