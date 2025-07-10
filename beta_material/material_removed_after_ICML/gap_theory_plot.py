import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def predicted_loss(
    t,  # time in years from t0
    B_inf,  # dollars per token, e.g. 1e-5  (=0.001¢)
    #
    # --- Baseline (t₀) & improvement rates ------------------------------
    flop_price0=1.15e-16,  # $ / FLOP at t0 (≈ A100 class)
    r_hw=1.30,  # $/FLOP gets 1.30× cheaper each year
    flop_per_p0=2.0,  # FLOP / param / token (dense fp16 Transformer)
    r_kernel=1.30,  # FLOP/param/token falls 1.30× each year
    r_train=2.8,  # "effective-parameter" gain per year (½-compute-in-8-mo rule)
    #
    # --- Size-only scaling-law fit (Epoch AI, 2024) ----------------------
    a=482.0,
    alpha=0.3478,
    b=1.8172,
):
    """
    Returns (loss_nats, loss_bits) for the given time t and B_inf.
    """
    # --- 1) $/FLOP(t) --------------------------------------------------------
    flop_price = flop_price0 / (r_hw**t)

    # --- 2) FLOP/param/token(t) ---------------------------------------------
    flop_per_p = flop_per_p0 / (r_kernel**t)

    # --- 3) Physical param budget -------------------------------------------
    P_max = B_inf / (flop_price * flop_per_p)

    # --- 4) Training-side "effectiveness" boost -----------------------------
    P_eff = P_max * (r_train**t)

    # --- 5) Scaling law → loss ----------------------------------------------
    loss_nats = b + a / (P_eff**alpha)
    loss_bits = loss_nats / np.log(2)

    return loss_nats, loss_bits


# Plot configuration
plt.figure(figsize=(12, 8))
t_values = np.linspace(0, 20, 100)  # Time from 0 to 20 years

# Plot for different B_inf values
B_inf_values = [1e-6, 1e-5, 1e-4, 1e-3]
colors = ["blue", "green", "red", "purple"]
labels = ["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$"]

for i, B_inf in enumerate(B_inf_values):
    loss_nats_values = []
    loss_bits_values = []

    for t in t_values:
        loss_nats, loss_bits = predicted_loss(t, B_inf)
        loss_nats_values.append(loss_nats)
        loss_bits_values.append(loss_bits)

    plt.plot(
        t_values,
        loss_bits_values,
        color=colors[i],
        linewidth=2,
        label=f"B_inf = {labels[i]} $/token",
    )

# Add plot elements
plt.xlabel("Time (years)", fontsize=14)
plt.ylabel("Loss (bits per token)", fontsize=14)
plt.title("Predicted Loss over Time for Different B_inf Values", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Use log scale for y-axis since loss values can span orders of magnitude
plt.yscale("log")
plt.gca().yaxis.set_major_formatter(ScalarFormatter())

# Add annotation explaining what B_inf represents
plt.figtext(
    0.5,
    0.01,
    "B_inf: Maximum dollars spent per token for inference\n"
    "Lower loss values indicate better model performance",
    ha="center",
    fontsize=10,
)

plt.tight_layout()
plt.savefig("predicted_loss_over_time.png", dpi=300)
plt.show()
