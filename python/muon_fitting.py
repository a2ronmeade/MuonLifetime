
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

filename = "../data/full_run"

# voltage * C = time
C = 1.957534           # calibation constant for period of 1.28 microseconds, us/V
sigma_C = 0.000141      # uncertainty, us/V

bin_width = 0.05        # voltage bin width (V)

data = np.loadtxt(filename)
data = data[3:] # ignore first few columns
voltage = data[:, 0]
counts = data[:, 1]


# Convert voltage to time
time = C * voltage

sigma_counts = np.maximum(np.sqrt(counts), 1.84) # Gehrels approximation thing for small N

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

# Calculate residuals and chi-squared
counts_model = exponential_decay(time, N0, tau, B)
residuals = counts - counts_model

chi_squared = np.sum((residuals / sigma_counts)**2)
dof = len(counts) - len(popt)
reduced_chi_squared = chi_squared / dof
normalized_residuals = residuals / sigma_counts

print(f"\nChi-squared:         {chi_squared:.3f}")
print(f"Degrees of freedom:  {dof}")
print(f"Reduced chi-squared: {reduced_chi_squared:.3f}")

sigma_N0_fit, sigma_tau_fit, sigma_B_fit = np.sqrt(np.diag(pcov))                   # statistical fit uncertainty
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
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 8),
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
)

# Main plot
ax1.bar(
    time,
    counts,
    width=C * bin_width,
    align="center",
    edgecolor="black",
    linewidth=0.3,
    label="Measured data"
)

ax1.errorbar(
    time,
    counts,
    yerr=sigma_counts,
    fmt="none",
    color="black",
    capsize=2,
    linewidth=0.8,
    label="Poisson uncertainty (Gehrel's approximation)"
)

ax1.plot(
    time_fit,
    counts_fit,
    color="red",
    linewidth=2,
    label="Exponential + background fit"
)

ax1.set_ylabel("Counts", size=20)
ax1.set_title("Exponential Decay with Background for Muon Decay", size=20)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=12)

# Fit information
fit_text = (
    rf"$N_0 = {N0:.1f} \pm {sigma_N0_total:.1f}$" "\n"
    rf"$\tau = {tau:.4f} \pm {sigma_tau_total:.4f}$ μs" "\n"
    rf"$B = {B:.2f} \pm {sigma_B_total:.2f}$ counts" "\n"
    rf"$\chi^2_\mathrm{{red}} = {reduced_chi_squared:.3f}$"
)

ax1.text(
    0.97, 0.95,
    fit_text,
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=14,
    bbox=dict(facecolor="white", alpha=0.85)
)

# Residuals plot
ax2.errorbar(
    time,
    normalized_residuals,
    yerr=np.ones_like(counts),
    fmt="o",
    markersize=3,
    color="black",
    capsize=2,
    linewidth=0.8
)

ax2.axhline(0, color="red", linewidth=1.5)
ax2.set_xlabel("Time (microseconds)", size=20)
ax2.set_ylabel("Normalized Residuals", size=16)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("exponential_decay_fit.png", dpi=300)
plt.show()
