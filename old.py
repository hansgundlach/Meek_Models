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
