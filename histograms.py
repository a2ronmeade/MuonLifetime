import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

os.makedirs("delay histograms", exist_ok=True)
df = pd.read_csv("delays.csv")

def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2)

for label in ['A', 'B', 'D']:
    subset = df[df['PMTlabel'] == label]

    if subset.empty:
        continue

    x = subset['delay'].to_numpy(dtype=float)
    y = subset['count'].to_numpy(dtype=float)


    p0 = [y.max(), np.average(x, weights=y), 15]
    popt, _ = curve_fit(gaussian, x, y, p0=p0)
    amplitude, mu, sigma = popt
    sigma = abs(sigma)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = gaussian(x_fit, amplitude, mu, sigma)

    # plot
    plt.figure(figsize=(8, 5))
    plt.bar(x, y, width=6, alpha=0.6, label='Data')
    plt.plot(
        x_fit, y_fit, 'r-', linewidth=2,
        label=(
            f'Gaussian fit\n'
            f'Mean = {mu:.2f} ns\n'
            f'Sigma = {sigma:.2f} ns\n'
            f'FWHM = {fwhm:.2f} ns'
        )
    )

    plt.xlabel('Delay (ns)')
    plt.ylabel('Count')
    plt.title(f'PMT {label}: Gaussian Fit')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save plot
    filename = f"delay histograms/histogram-{label.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Saved {filename}")