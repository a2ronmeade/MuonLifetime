
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

filename = "muon_run_1"

# voltage * C = time
C = 0.978767            # calibation constant for if our period is actually 0.64 microseconds (needs to be checked)
#C = 0.489384           # us/V
sigma_C = 0.000035      # us/V

bin_width = 0.05        # voltage bin width (V)

data = np.loadtxt(filename)
data = data[2:] # ignore first few columns
voltage = data[:, 0]
counts = data[:, 1]


# Convert voltage to time
time = C * voltage

sigma_counts = np.sqrt(np.maximum(counts, 1)) # Poisson uncertainty in each bin
sigma_V = bin_width / np.sqrt(12)
sigma_time_bin = C * sigma_V

sigma_time_cal = sigma_C * np.abs(voltage) # Convert calibration uncertainty to time uncertainty

# Exponential decay model
def exponential_decay(t, N0, tau, B):
    return N0 * np.exp(-t / tau) + B

N0_guess = np.max(counts)
tau_guess = (np.max(time) - np.min(time)) / 3
B_guess = np.min(counts)

# Weighted exponential fit using Poisson uncertainties
popt, pcov = curve_fit(
    exponential_decay,
    time,
    counts,
    p0=[N0_guess, tau_guess, B_guess],
    sigma=sigma_counts,
    absolute_sigma=True,
    bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
    maxfev=10000
)
N0, tau, B = popt

sigma_N0_fit, sigma_tau_fit, sigma_B_fit = np.sqrt(np.diag(pcov))                                # statistical fit uncertainty
sigma_tau_bin = tau * (sigma_V / np.mean(np.abs(voltage)))                          # binning uncertainty
sigma_tau_cal = tau * sigma_C / C                                                   # scale uncertainty proportionally
sigma_tau_total = np.sqrt(sigma_tau_fit**2 + sigma_tau_bin**2 + sigma_tau_cal**2)   # add in quad 

sigma_N0_total = sigma_N0_fit
sigma_B_total = sigma_B_fit

print(f"N0  = {N0:.3f} +/- {sigma_N0_total:.3f} counts")
print(f"tau = {tau:.5f} +/- {sigma_tau_total:.5f} us")
print(f"B = {B:.3f} +/- {sigma_B_total:.3f} counts")

print("\n--- Uncertainty breakdown ---")
print(f"N0 fit uncertainty:          {sigma_N0_fit:.5f}")
print(f"Tau fit uncertainty:         {sigma_tau_fit:.5f} us")
print(f"Tau discretization error:    {sigma_tau_bin:.5f} us")
print(f"Tau calibration uncertainty: {sigma_tau_cal:.5f} us")
print(f"Time bin uncertainty:        {sigma_time_bin:.5f} us")

# generate fit curve
time_fit = np.linspace(np.min(time), np.max(time), 1000)
counts_fit = exponential_decay(time_fit, N0, tau, B)

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
    counts_fit,
    color="red",
    linewidth=2,
    label="Exponential + background fit"
)

plt.xlabel("Time (us)")
plt.ylabel("Counts")
plt.title("Exponential Decay + Background Fit")
plt.grid(alpha=0.3)
plt.legend()

fit_text = (
    rf"$N_0 = {N0:.1f} \pm {sigma_N0_total:.1f}$" "\n"
    rf"$\tau = {tau:.4f} \pm {sigma_tau_total:.4f}$ us" "\n"
    rf"$B = {B:.2f} \pm {sigma_B_total:.2f}$ counts"
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
plt.savefig("exponential_decay_fit.png", dpi=300)
plt.show()
