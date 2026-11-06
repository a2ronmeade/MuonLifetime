
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

filename = "muon_run_1"

# Voltage * C = time
C = 0.489384             # us/V
sigma_C = 0.000035       # us/V

bin_width = 0.05         # V

# Assumed muon fractions
fraction_plus = 0.55
fraction_minus = 0.45

# Load data
data = np.loadtxt(filename)
data = data[2:]          # Ignore first two bins

voltage = data[:, 0]
counts = data[:, 1]

# Convert voltage to time
time = C * voltage

# Poisson uncertainties
sigma_counts = np.sqrt(np.maximum(counts, 1))

# Discretization uncertainty
sigma_V = bin_width / np.sqrt(12)
sigma_time_bin = C * sigma_V

# Exponential mixture model
# We parameterize tau_plus = tau_minus + delta_tau
# to ensure tau_plus > tau_minus.

def double_exponential(t, N0, tau_minus, delta_tau, B):

    tau_plus = tau_minus + delta_tau

    return (
        N0 * (
            fraction_plus * np.exp(-t / tau_plus)
            + fraction_minus * np.exp(-t / tau_minus)
        )
        + B
    )


# Initial guesses
N0_guess = np.max(counts)
tau_minus_guess = 2.026
delta_tau_guess = 2.197 - 2.026
B_guess = np.mean(counts[-20:])

# Weighted fit
popt, pcov = curve_fit(
    double_exponential,
    time,
    counts,
    p0=[
        N0_guess,
        tau_minus_guess,
        delta_tau_guess,
        B_guess
    ],
    sigma=sigma_counts,
    absolute_sigma=True,
    bounds=(
        [0, 0.1, 0.0001, 0],
        [np.inf, 5, 5, np.inf]
    ),
    maxfev=50000
)

N0, tau_minus, delta_tau, B = popt
tau_plus = tau_minus + delta_tau

# Fit uncertainties
sigma_N0 = np.sqrt(pcov[0, 0])
sigma_tau_minus = np.sqrt(pcov[1, 1])
sigma_delta = np.sqrt(pcov[2, 2])
sigma_B = np.sqrt(pcov[3, 3])

# Propagate covariance into tau_plus
sigma_tau_plus = np.sqrt(
    pcov[1, 1] + pcov[2, 2] + 2 * pcov[1, 2]
)

# Calibration uncertainty
sigma_tau_minus_cal = tau_minus * sigma_C / C
sigma_tau_plus_cal = tau_plus * sigma_C / C

# Combine fit and calibration uncertainties
sigma_tau_minus_total = np.sqrt(
    sigma_tau_minus**2 + sigma_tau_minus_cal**2
)

sigma_tau_plus_total = np.sqrt(
    sigma_tau_plus**2 + sigma_tau_plus_cal**2
)

# Print results
print("\n--- Double Exponential Fit ---")

print(f"N0        = {N0:.3f} +/- {sigma_N0:.3f} counts")

print(
    f"Tau plus  = {tau_plus:.4f} "
    f"+/- {sigma_tau_plus_total:.4f} us"
)

print(
    f"Tau minus = {tau_minus:.4f} "
    f"+/- {sigma_tau_minus_total:.4f} us"
)

print(f"Background = {B:.3f} +/- {sigma_B:.3f} counts")

print("\n--- Uncertainty Breakdown ---")

print(f"Tau plus fit uncertainty:  {sigma_tau_plus:.5f} us")
print(f"Tau minus fit uncertainty: {sigma_tau_minus:.5f} us")

print(f"Tau plus calibration:      {sigma_tau_plus_cal:.5f} us")
print(f"Tau minus calibration:     {sigma_tau_minus_cal:.5f} us")

print(f"Time bin uncertainty:      {sigma_time_bin:.5f} us")


# Generate fit curves
time_fit = np.linspace(np.min(time), np.max(time), 1000)

tau_plus_fit = tau_plus
tau_minus_fit = tau_minus

plus_component = (
    N0 * fraction_plus * np.exp(-time_fit / tau_plus_fit)
)

minus_component = (
    N0 * fraction_minus * np.exp(-time_fit / tau_minus_fit)
)

background = np.full_like(time_fit, B)

total_fit = plus_component + minus_component + background


# Plot
plt.figure(figsize=(12, 6))

plt.bar(
    time,
    counts,
    width=C * bin_width,
    align="center",
    edgecolor="black",
    linewidth=0.3,
    label="Measured data"
)

plt.errorbar(
    time,
    counts,
    yerr=sigma_counts,
    fmt="none",
    color="black",
    capsize=2,
    linewidth=0.8,
    label="Poisson uncertainty"
)

plt.plot(
    time_fit,
    total_fit,
    color="red",
    linewidth=2.5,
    label="Total fit"
)

plt.plot(
    time_fit,
    plus_component + B,
    "--",
    linewidth=1.5,
    label="Positive muon component + background"
)

plt.plot(
    time_fit,
    minus_component + B,
    "--",
    linewidth=1.5,
    label="Negative muon component + background"
)

plt.axhline(
    B,
    color="gray",
    linestyle=":",
    label="Background"
)

plt.xlabel("Time (us)")
plt.ylabel("Counts")
plt.title("Double Exponential Muon Decay Fit")

plt.grid(alpha=0.3)
plt.legend()

fit_text = (
    rf"$N_0 = {N0:.1f} \pm {sigma_N0:.1f}$" "\n"
    rf"$\tau_+ = {tau_plus:.3f} \pm {sigma_tau_plus_total:.3f}$ us" "\n"
    rf"$\tau_- = {tau_minus:.3f} \pm {sigma_tau_minus_total:.3f}$ us" "\n"
    rf"$B = {B:.2f} \pm {sigma_B:.2f}$"
)

plt.text(
    0.97, 0.95,
    fit_text,
    transform=plt.gca().transAxes,
    ha="right",
    va="top",
    bbox=dict(facecolor="white", alpha=0.85)
)

plt.tight_layout()
plt.savefig("double_exponential_decay_fit.png", dpi=300)
plt.show()