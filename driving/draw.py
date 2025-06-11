import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
'''
Env: sp_v2_new
'''
path = root = '/home/jzhang2297/data/udacity_output/testing/'
ys=[]
with open(path + 'CH2_final_evaluation.csv', 'r') as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        ys.append(float(line.split(',')[1]))

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(ys, bins=10, edgecolor='black')  # Adjust bins as needed
plt.title("Histogram of 'ys'")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Save the plot
plot_path = "density_plot_ys.png"
plt.savefig(plot_path)
plt.close()