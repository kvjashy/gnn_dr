import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



sns.set_theme()

# Path to data
input_paths = [


               ]


# Load the data
lists = []
for path in input_paths:
    with open(os.path.join('runs', path, 'rollout_mean.txt')) as file:
        rewards = [float(line.strip('[').strip(']\n')) for line in file.readlines()]
        lists.append(rewards)

# Find the length of the smallest dataset among all
min_len = min([len(reward_list) for reward_list in lists])

# Scale all datasets to the length of the smallest one
scaled_lists = [reward_list[:min_len] for reward_list in lists]

# Smooth the data using moving average
window_size = 50  # Adjust as needed
smoothed_lists = [np.convolve(reward_list, np.ones(window_size)/window_size, mode='valid') for reward_list in scaled_lists]

# Group the smoothed datasets: every three datasets
grouped_lists = [smoothed_lists[i:i+3] for i in range(0, len(smoothed_lists), 3)]

# Plotting
plt.figure(dpi=300)

labels = []  # Added labels for new averages
colors = ["blue", "red", "green", "purple"]  # Added colors for new averages

for idx, group in enumerate(grouped_lists):
    averages = np.mean(group, axis=0)
    std_devs = np.std(group, axis=0)
    
    plt.plot(averages, label=labels[idx], color=colors[idx])
    plt.fill_between(range(len(averages)), averages-std_devs, averages+std_devs, color=colors[idx], alpha=0.2)

plt.xlabel('number rollouts with 1024 steps per rollout (smoothed)')
plt.ylabel('mean reward in rollout')
plt.legend()
plt.title('Nervenet and dynamic dropout(4 and 6 layers) ')
plt.savefig('6layers.png')
plt.show()
plt.clf()


