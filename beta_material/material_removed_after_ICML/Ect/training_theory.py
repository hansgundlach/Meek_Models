# # %%
# # model of training overhang over time
# import numpy as np
# import matplotlib.pyplot as plt
# import sympy as sp
# from sympy import latex

# #%%

# cost = 1000
# cost_per_flop_year = 0.8
# flops_per_dollar_year0 = 1e9 / 0.02
# alg_gains_train = 1.8
# alg_gains = alg_gains_train
# #chinchilla function relating compute to perplexity  
# chin_func = lambda x: 1070 * x ** (-0.154) + 1.7
# total_loss = lambda cost, time: chin_func(
#     (alg_gains**time) * flops_per_dollar_year0 * cost / ((cost_per_flop_year) ** time)
# )


# #%%
# # now graph this function with respect to time
# time = np.linspace(0,100, 100)
# player1_growth = 1.5
# player2_growth = 1.1
# # loss for best model over time
# player2loss = total_loss(1000 * (player2_growth) ** time, time)
# player1loss= total_loss(1000 * (player1_growth) ** time, time)
# logistic_investment = lambda time: 10000/ (1 + np.exp(-0.1 * (time-10)))
# logistic_loss = total_loss(logistic_investment(time), time)


# # best peplexity loss in the world 
# world_loss = lambda time:  total_loss(1000 * (player1_growth) ** time, time)
# # graph with multiple costs adding cost labels
# plt.figure(figsize=(10, 5))
# for c in [1000, 10000, 100000]:
#     loss = total_loss(c, time)
#     plt.plot(time, player1loss, label="Exponential Investment")
#     plt.plot(time, loss, label=f"Cost: {c}")
#     plt.plot(time, logistic_loss,label="Logistic Investment")
#     # plt.plot(time, loss, label=f"Cost: {c}")
# plt.xlabel("Time (years)")
# plt.ylabel("Total Loss")
# plt.legend()
# plt.title("Training Loss Over Time")
# plt.grid(True)
# # plt.yscale("log")

# #%%


# #plot of overhang over time 
# # plt.figure(figsize=(10, 5))
# plt.plot(time, player1loss/total_loss(1000, time), label="Overhang over constant player")
# plt.plot(time, player1loss/player2loss, label="Second Best Model")
# plt.plot(time, logistic_loss/total_loss(1000, time), label="Overhang over logistic investment")
# plt.xlabel("Time (years)")
# plt.ylabel(" Perplexity Overhang")
# plt.legend()
# plt.title("Training Overhang Over Time")

# # %%


# #modle of inference overhang with set number of parameters
# # ====================================================================================================

# # model of inference overhang
# player1_growth = 1.9
# player2_growth = 1.0
# alg_gains_inf = 1.2



# # this is total total loss using inference with squared computer 
# # total_loss_inf = lambda cost, time: chin_func(
# #     (
# #         (alg_gains**time)
# #         * flops_per_dollar_year0
# #         * cost
# #         / ((cost_per_flop_year) ** time)
# #     )
# #     ** 2
# # )



# total_loss_inf = lambda cost, time: chin_func(((cost/(cost_per_flop_year**time))**2 * (alg_gains_train**time) * flops_per_dollar_year0))


# time = np.linspace(0, 40, 10000)
# for c in [1000, 10000, 100000]:
#     loss = total_loss(c, time)
#     plt.plot(time, loss, label=f"Cost: {c}")


# # best peplexity loss in the world 
# best_loss = lambda time:  total_loss(10000000 * (player1_growth) ** time, time)
# best_loss_seq = world_loss(time)
# inf_overhang = total_loss_inf(100000, time) / total_loss_inf(100, time)
# inf_bestvs1000dollars =best_loss_seq / total_loss_inf(1, time)
# # inf_overhang_variable = total_loss_inf((player1_growth) ** time, time) / total_loss_inf(
# #     (player2_growth) ** time, time
# # )
# plt.ylabel("Total Loss for Models ")
# plt.xlabel("Time (years)")
# plt.title("Inference Overhang for Given Inference Cost")
# plt.legend()



# plt.figure(figsize=(10, 5))
# plt.plot(time, inf_overhang, "r", label="Inference Overhang 100000 dollar model vs 100 dollar model")
# plt.plot(time, inf_bestvs1000dollars, label="Inference Overhang Best Model vs 1000 dollar model")
# # plt.plot(time, inf_overhang_variable, label="Inference Overhang Variable Players growth1/growth2")
# plt.title("Inference Overhang of 10000 dollar model vs 100 dollar model Over Time")
# plt.xlabel("Time (years)")
# plt.ylabel("Overhang of Inference")
# plt.legend()
# plt.grid(True)

# # inference overhang model
# # this could model US china embargo relations
# # this could also model two companies asking when to aquire
# #%%
# #now plot inference ovehang and training overhang at the same time 

# # plt.plot(time, player1loss/total_loss(1000, time), label="Overhang over constant player")

# # compute the training overhang 
# player1loss= total_loss(1000 * (player1_growth) ** time, time)
# plt.figure(figsize=(10, 5))
# plt.plot(time, inf_bestvs1000dollars, label="Inference Overhang")
# plt.plot(time, player1loss/total_loss(1000, time), label="Training Overhang")
# plt.xlabel("Time (years)")
# plt.ylabel("Overhang")
# plt.legend()
# plt.title("Inference and Training Overhang Over Time")



# #%%

# #inference overhang for constant parameter models
# time = np.linspace(0, 100, 10000)
# cost_per_flop_year = 1
# # loss_chin_inf = lambda cost, time: chin_func((alg_gains_inf**time)*(alg_gains_train**time) * flops_per_dollar_year0 * cost / ((cost_per_flop_year) ** time)
# # )
# loss_constant = 1.7
# const_params = 1
# A = 1
# loss_const_param = lambda time: loss_constant + A/(const_params*alg_gains**time)**0.34
# plt.figure(figsize=(10, 5))
# # set y axis scale from 0 to 5
# # plt.ylim(0, 5)
# plt.xlabel("Time (years)")
# plt.ylabel("Inference Overhang")
# world_loss_seq = world_loss(time)
# loss_infoptimal_year_seq = loss_const_param(time)
# plt.plot(time, world_loss_seq/loss_infoptimal_year_seq, label="Inference Overhang")
# plt.title("world_best/const_parm Over Time")
# plt.legend()


# # %%
# # another model of inference overhang this is assuming fixed param for better and base model 
# time = np.linspace(0, 40, 10000)
# plt.figure(figsize=(10, 5))
# plt.plot(time, 1.5 ** (-0.034 * time), label="Inference Overhang")

# # %%
# # symbolic computations for overhang

# # define the variables
# cost, time = sp.symbols("cost time")
# # define more varaibles
# g_alg= sp.symbols("g_alg")
# C0 = sp.symbols("C0") # flops per dollar year 0
# I = sp.symbols("I") # investment
# t = sp.symbols("t") # time
# g_flop = sp.symbols("g_flop") # growth in flops per dollar per year
# g_invest = sp.symbols("g_invest") # growth in investment per year 

# # define the function
# chin_func = 1070 * I ** (-0.154) + 1.7
# # display function 
# total_loss = chin_func.subs(I, (g_alg**t) * C0 * I * g_flop**t)
# # display the function
# total_loss
# #simplify the function
# total_loss = sp.simplify(total_loss)
# #overhang amount  
# overhang = total_loss.subs(I, (g_alg**t)*g_invest**t * C0 * I * g_flop**t) / total_loss.subs(I, C0 * I * (g_alg**t)*g_flop**t)

# overhang = sp.simplify(overhang)
# # display the simplified function
# print(latex(total_loss))

# sp.pprint(total_loss)
# #the overhang is 
# print("the overhang is")
# sp.pprint(overhang)
# #simplify exponential
# total_loss = sp.simplify(total_loss)

# # %%
