import matplotlib.pyplot as plt
import numpy as np

def plot_weibull_polar(data, height=100):
    if height not in data:
        print(f"Height {height}m not in dataset.")
        return

    values = data[height]
    num_dirs = len(values)
    angles = np.deg2rad(np.arange(0, 360, 360 // num_dirs))

    f_vals = [x[0] for x in values]
    A_vals = [x[1] for x in values]
    k_vals = [x[2] for x in values]

    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(18, 6))

    axs[0].bar(angles, f_vals, width=2*np.pi/num_dirs, bottom=0.0)
    axs[0].set_title(f"Sector Frequency (f) at {height}m")

    axs[1].bar(angles, A_vals, width=2*np.pi/num_dirs, bottom=0.0)
    axs[1].set_title(f"Weibull Scale (A) at {height}m")

    axs[2].bar(angles, k_vals, width=2*np.pi/num_dirs, bottom=0.0)
    axs[2].set_title(f"Weibull Shape (k) at {height}m")

    for ax in axs:
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    plt.tight_layout()
    plt.show()

from gwa_parser import gwa3_data_parser

data = gwa3_data_parser("gwa3_HornsRev1_gwc.lib")

plot_weibull_polar(data, height=150) 