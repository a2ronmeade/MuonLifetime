import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("../data/time_calibration")
voltage = data[:, 0]
counts = data[:, 1]

T = 1.28  # period in microseconds (we need to check this)
T_uncertainty = T * 0.00005  # given instrument uncertainty?
bin_width = 0.05

# assign locations of the peaks (from visual inspection)
peak_centers = np.array([
    0.625, 1.325, 1.975, 2.625, 3.275,
    3.925, 4.575, 5.225, 5.875, 6.575,
    7.225, 7.875, 8.525, 9.175, 9.825
])

peak_numbers = np.arange(1, len(peak_centers) + 1)
mask = counts > 0
voltage_nonzero = voltage[mask]
counts_nonzero = counts[mask]
nearest_peak = np.argmin(np.abs(voltage_nonzero[:, None] - peak_centers), axis=1)
distance = np.min(np.abs(voltage_nonzero[:, None] - peak_centers), axis=1)

# allow spillover
valid = distance <= 0.051
voltage_assigned = voltage_nonzero[valid]
counts_assigned = counts_nonzero[valid]
n_assigned = peak_numbers[nearest_peak[valid]]

# are bins valid?
if np.any(~valid):
    print("Unassigned bins:")
    print(voltage_nonzero[~valid])

voltage_per_period = voltage_assigned / n_assigned

# Count-weighted average & weighted uncertainty
N_total = np.sum(counts_assigned)
X_mean = np.average(voltage_per_period, weights=counts_assigned)
variance = np.average((voltage_per_period - X_mean)**2, weights=counts_assigned)
sigma_X = np.sqrt((variance + bin_width**2 / 12) / N_total)


# calibration constant and uncertainty
C = T / X_mean
sigma_C = C * np.sqrt((T_uncertainty / T)**2 + (sigma_X / X_mean)**2)

print(f"Total counts: {N_total:.0f}")
print(f"Voltage per period: {X_mean:.6f} +/- {sigma_X:.6f} V")
print(f"Period T: {T:.4f} us")
print(f"Calibration constant: {C:.6f} +/- {sigma_C:.6f} us/V")

print(f"\nTime = Voltage * ({C:.6f} us/V)")


# histogram plot
plt.figure(figsize=(12, 6))
plt.bar(
    voltage, counts,
    width=bin_width,
    align="center",
    edgecolor="black",
    linewidth=0.3
)

plt.xlabel("Voltage (V)", size = 14)
plt.ylabel("Counts", size = 14)
plt.title("Count v. Voltage from the Time Calibration Device", size = 18)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("time_cal_plot.png", dpi=300, bbox_inches="tight")
plt.show()