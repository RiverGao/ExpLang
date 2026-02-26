# draw multiple win-tie-lose bars in one chart
# 1. 0.2162-0.5928-0.1910
# 2. 0.1578-0.5085-0.3337
# 3. 0.1053-0.6277-0.2670
# 2-2. 0.1716-0.5509-0.2775
# 3-2. 0.1295-0.6256-0.2449


from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

# labels = ['Qwen3-4B', '+SFT (Ours)', '+SFT+RL (Ours)']
# win = [0.2162, 0.1578, 0.1053]
# tie = [0.5928, 0.5085, 0.6277]
# lose = [0.1910, 0.3337, 0.2670]

# labels = ['Qwen3-4B', '+SFT (Ctrl)', '+SFT+RL (Ctrl)']
# win = [0.2162, 0.1716, 0.1295]
# tie = [0.5928, 0.5509, 0.6256]
# lose = [0.1910, 0.2775, 0.2449]

labels = ['Qwen3-4B', 'Controlled', 'Ours']
win = [0.2162, 0.1295, 0.1053]
tie = [0.5928, 0.6256, 0.6277]
lose = [0.1910, 0.2449, 0.2670]

width = 0.35  # the width of the bars
y = 1.25 * width * np.arange(len(labels))  # the label locations
fig, ax = plt.subplots(figsize=(5.5, 2.5))
bars1 = ax.barh(y, win, width, label='Win', color='#5DADE2')
bars2 = ax.barh(y, tie, width, left=win, label='Tie', color='#F4D03F')
bars3 = ax.barh(y, lose, width, left=np.array(win) + np.array(tie), label='Lose', color='#E74C3C')

# Set font and size
fp = fm.FontProperties(fname='assets/Times New Roman.ttf')
plt.rcParams['font.size'] = 16

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Proportion', fontproperties=fp)
ax.set_title('Win-Tie-Lose Proportions of Non-English vs. English', fontproperties=fp)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontproperties=fp)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), prop=fp, ncol=3)
ax.set_xlim(0, 1)
ax.xaxis.set_ticks([])  # Remove x-ticks
ax.bar_label(bars1, fmt='%.2f%%', padding=3, label_type='center', fontproperties=fp, labels=[f'{v*100:.2f}%' for v in win])
ax.bar_label(bars2, fmt='%.2f%%', padding=3, label_type='center', fontproperties=fp, labels=[f'{v*100:.2f}%' for v in tie])
ax.bar_label(bars3, fmt='%.2f%%', padding=3, label_type='center', fontproperties=fp, labels=[f'{v*100:.2f}%' for v in lose])

fig.tight_layout()
plt.savefig('plot/results/win-tie-lose-main.pdf', dpi=300)
# plt.show()
