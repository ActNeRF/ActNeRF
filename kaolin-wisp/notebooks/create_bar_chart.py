import matplotlib.pyplot as plt
import numpy as np

# Data
without_flipping = [0.571428571, 0.142857143, 0.285714286]
with_flipping = [0.714285714, 0.071428571, 0.214285714]
labels = ['Success', 'Pose Estimation Failure', 'Grasp Failure']
colors = ['#039BE5', '#FF5733', '#FFC154']

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Set positions for bars
x = np.arange(len(labels))
bar_width = 0.35

# Without flipping
ax.bar(x - bar_width/2, without_flipping, bar_width, label='Without Flipping', color=colors[0])
# With flipping
ax.bar(x + bar_width/2, with_flipping, bar_width, label='With Flipping', color=colors[1])

# Set labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Proportion')
ax.set_title('Comparison of Success Rate with and without Flipping')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Show plot
plt.tight_layout()
# plt.show()
plt.savefig('bar1.png')
