
#%%
#import evertything necessary
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates



# %%
# basic fuzzer model


# I want to make a multivariate function
def multi_func(g):
    risk = 0
    N = 100
    at_each_year = []
    # attack = lambda x: N*(1-(1-(1/N))**x)
    # defense = lambda x: N*(1-(1-(1/N))**x)
    for t in range(0, 10):
        risk += min(g**t, 1000) * (1000 - min((g**t) * 10, 1000)) / 1000
    return risk
# %%
# used a model for red teaming language models
# generation of code and not
# variations what if we want to minimize the chance of one error
# what if we want to minimize the probablity of an individual finding an error
# what if we want to publish a technology but loose out if we wait too long
# what is the risk if we do search in parallel
growth_vulnerabilities = 1.2
def coupon_total(g, alpha=3, growth_vulnerabilities=growth_vulnerabilities):
    risk = 0
    at_each_year = []
    attack = lambda x: (x) ** (1 / alpha)
    defense = lambda x: (x) ** (1 / alpha)
    for t in range(0, 20):
        N = 100 * (growth_vulnerabilities) ** t
        year_t = min(attack(g**t), N) * (N - min(defense(10 * g**t), N)) / N
        at_each_year.append(year_t)
        risk += year_t
    return risk, at_each_year
#%%

# do this for a variety of growth rates
growth_rate_1 = 25
risk, at_each_year = coupon_total(growth_rate_1)
# plot the function over a range of value
plt.plot(at_each_year, label=f"g = {growth_rate_1}")


growth_rate2 = 1.2
risk, at_each_year = coupon_total(growth_rate2)
plt.plot(at_each_year, label=f"g = {growth_rate2}")
# risk, at_each_year = coupon_total(1.4)
# plt.plot(at_each_year)

plt.title("Risk Over Time")
plt.xlabel("Year")
plt.ylabel("Zero Days Found")
plt.legend()
plt.grid()
plt.show()
#%%

g = np.arange(1.0001, 6, 0.01)
output = [coupon_total(g)[0] for g in g]
plt.plot(g/growth_vulnerabilities, output)
plt.xlabel("Growth Rate of Technology, relative to vulnerabilities")
plt.ylabel("Zero Day Found")
plt.grid()
plt.title("Total Risk Based on Growth Using Power Law Coupon")
#%%
growth_vulnerabilities = 1.2
def coupon_cummulative(g):
    alpha = 2
    risk = 0
    at_each_year = []
    attack = lambda x: (x) ** (1 / alpha)
    defense = lambda x: (x) ** (1 / alpha)
    for t in range(0, 20):
        N = 100 * (growth_vulnerabilities) ** t
        year_t = min(attack(g**t), N) * (N - min(defense(10 * g**t), N)) / N
        at_each_year.append(year_t)
        risk += year_t
    return risk, at_each_year
# %%
# repalce by coupon collector problem
# ie number of unique vulnerabilites found afters fuzzing n times:
def coupon_collector(g):
    risk = 0
    N = 100
    attack = lambda x: N * (1 - (1 - (1 / N)) ** x)
    defense = lambda x: N * (1 - (1 - (1 / N)) ** x)
    for t in range(0, 100):
        risk += min(attack(g**t), N) * (N - min(defense(g**t), N)) / N
    return risk


# what about fixed size N vulnerabilites accumpulated over time


def cummulative(g):
    risk = 0
    attack = 0
    defense = 0
    for t in range(0, 100):
        risk += min(attack, 1000) * (1000 - min((defense, 1000))) / 1000
        attack += g**t
        defense += g**t
    return risk


# look at risk over time

# graph this function
import numpy as np
import matplotlib.pyplot as plt

# plot the function over a range of values
g = np.arange(1.0001, 2, 0.01)
output = [coupon_collector(g) for g in g]
plt.plot(g, output)
plt.xlabel("Growth Rate of Technology")
plt.ylabel("Zero Day Found")
plt.grid()

# %%

# now look at cummulant
g = np.arange(1.0001, 2, 0.01)
output = [cummulative(g) for g in g]
plt.plot(g, output)
plt.xlabel("Growth Rate of Technology")
plt.ylabel("Zero Day Found")
plt.grid()


# g = np.arange(0, 2, 0.1)
# plt.plot(g, multi_func(g))
# plt.grid()


# %%


def risk_time_t(t):
    total_vuln = 5 * t
    attack = 3 * t
    defense = 3 * t
    risk = (
        min(total_vuln, attack) * (total_vuln - min(defense, total_vuln)) / total_vuln
    )
    return risk


# plot the function over a range of values
t = np.arange(0, 10, 0.01)
output = [risk_time_t(t) for t in t]

plt.plot(t, output)

# %%
# predicting bounds on number of agents
budget = 100
pwin_many = lambda n: 1 - np.exp(-1 * (3 - np.log((budget / n) ** 2))) ** n
agents = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pwin = [pwin_many(n) for n in agents]
plt.plot(agents, pwin)


# how would I choose max inference
# ie shoudl I pay for more tokens/longer COT or just get a larger model

# could I get to AGI with just inference


# %%
