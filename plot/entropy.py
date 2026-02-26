import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# plot the entropy curves of ours vs. controlled
# data path: plot/results/entropy_rl.csv
# data format:
# Step,Controlled,Ours,
# 1,0.311599791,0.427150458,
# 2,0.313811511,0.414924562,
# 3,0.310405672,0.410435975,
# 4,0.309353799,0.409299791,
# 5,0.310773075,0.408585429,
# 6,0.308080733,0.397612303,
# 7,0.303926915,0.396570951 ...

# Set font and size
fp = fm.FontProperties(fname='assets/Times New Roman.ttf')
plt.rcParams['font.size'] = 16

# Load data
df = pd.read_csv('plot/results/entropy_rl.csv')

# Plot
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(df['Step'], df['Controlled'],'--', label='Controlled', linewidth=2, color='grey')
ax.plot(df['Step'], df['Ours'], '-', label='Ours', linewidth=2, color='green')

# Labels and title
ax.set_xlabel('RL Step', fontproperties=fp)
ax.set_ylabel('Entropy', fontproperties=fp)
ax.set_title('Entropy during RLVR', fontproperties=fp)
ax.legend(fontsize=11, prop=fp)
ax.grid(True, alpha=0.3)

# Formatting ticks
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontproperties(fp)

# Save and show
plt.tight_layout()
plt.savefig('plot/results/entropy_plot.pdf', dpi=300)
# plt.show()


