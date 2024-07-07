import matplotlib.pyplot as plt

# Data
without_flipping = [0.571428571, 0.142857143, 0.285714286]
with_flipping = [0.714285714, 0.071428571, 0.214285714]
labels = ['Success', 'Pose Estimation Failure', 'Grasp Failure']

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(6, 4))
colors = ['#039BE5', '#FF5733', '#FFC154']

# Without flipping
wedges1, _, autotexts1 = ax[0].pie(without_flipping, labels=None, autopct='%1.1f%%', startangle=90, wedgeprops={'linewidth': 2, 'edgecolor': 'white'}, colors=colors, radius=1.2)
ax[0].set_title('Without Flipping', fontsize=16)

# With flipping
wedges2, _, autotexts2 = ax[1].pie(with_flipping, labels=None, autopct='%1.1f%%', startangle=90, wedgeprops={'linewidth': 2, 'edgecolor': 'white'}, colors=colors, radius=1.2)
ax[1].set_title('With Flipping', fontsize=16)

# Legend
legend = fig.legend(wedges1, labels, loc='lower center', ncol=3, fontsize='large')
plt.setp(legend.get_texts(), fontsize='large')

# Increase font size of values
for autotext in autotexts1 + autotexts2:
    autotext.set_fontsize(10)

# Adjust layout
plt.subplots_adjust(
  # left = 0.0,
  # right = 1.0, 
  # top = 1.0, 
  # bottom=0.1, 
  wspace=0, 
  hspace=0
)
plt.tight_layout()
plt.show()

plt.savefig('pie2.png')