import math
import numpy as np
import matplotlib.pyplot as plt

# Data
labels = ['Success', 'Grasp Failure', 'Pose Estimation Failure']
categories = ['Active', 'Active+S', 'Active+S+F', 'Ours - F', 'Ours']
success = 100*np.array([0, 0.357142857, 0.571428571, 0.571428571, 0.714285714])
pose_failure = 100*np.array([1, 0.142857143, 0.142857143, 0.142857143, 0.071428571])
grasp_failure = 100*np.array([0, 0.5, 0.285714286, 0.285714286, 0.214285714])

colors = ['#039BE5', '#FF5733', '#FFC154']

# Create subplots
fig = plt.figure(figsize=(9, 5), dpi=300)
ax = plt.subplot()

# Stacked bar chart
bar_width = 0.5
index = np.arange(len(categories))
opacity = 1

bar1 = ax.bar(index, success, bar_width, label='Success', color=colors[0], alpha=opacity)
bar3 = ax.bar(index, grasp_failure, bar_width, bottom=np.array(success), label='Grasp Failure', color=colors[2], alpha=opacity)
bar2 = ax.bar(index, pose_failure, bar_width, bottom=np.array(success) + np.array(grasp_failure), label='Pose Estimation Failure', color=colors[1], alpha=opacity)


# Add labels, title and legend
# ax.set_xlabel('Method', fontsize=20)
ax.set_ylabel('Fraction of Instances (in %)', fontsize=20)
# ax.set_title('Success, Pose Estimation Failure, and Grasp Failure')
ax.set_xticks(index)
plt.yticks(fontsize=18)
ax.set_xticklabels(categories, rotation=0, fontsize=18)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=17)
ax.set_ylim(0,100)

# Show plot
plt.tight_layout()
# plt.show()

# for r1, r2, r3 in zip(bar1, bar2, bar3):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     h3 = r2.get_height()

#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1, ha="center", va="bottom", color="white", fontsize=16, fontweight="bold")
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="bottom", color="white", fontsize=16, fontweight="bold")
#     plt.text(r3.get_x() + r3.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="bottom", color="white", fontsize=16,
#     fontweight="bold") 

for c in ax.containers:
    # Optional: if the segment is small or 0, customize the labels
    labels = [int(round(v.get_height(),0)) if v.get_height() > 0 else '' for v in c]
    # remove the labels parameter if it's not needed for customized labels
    ax.bar_label(c, labels=labels, label_type='center', fontsize=15,
    # fontweight='bold', 
    color='black')

plt.savefig('bar1.pdf', format='pdf')