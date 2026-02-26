# data
# 1. accuracy
# dataset, en, multi, self
# math-500, 79.35, 75.67, 78.12
# aime2025, 18.61, 16.67, 18.89
# olymmath, 6.58, 5, 7
#
# 2. pass@k
# dataset, en, multi, self
# math-500, 91.6, 93.4, 90
# aime2025, 33.33, 30, 30
# olymmath, 19, 25, 22
# 
# 3. thinking length
# dataset, en, multi, self
# math-500, 2900.67, 2339.74, 2953.22
# aime2025, 4056.24, 3966.70, 4047.35
# olymmath, 4084.87, 4065.73, 4079.80
#
# 4. language compliance rate
# dataset, multi
# math-500, 73.28
# aime2025, 71.94
# olymmath, 74.67
#
# plots
# 1. accuracy and pass@k charts
# x axis: datasets, each dataset has three settings (en, multi, self)
# left y axis: accuracy (%)
# right y axis: pass@k (%)
# accuracy: bar chart
# pass@k: line chart
# 2. thinking length and language compliance rate charts
# x axis: datasets, each dataset has three settings (en, multi, self)
# left y axis: thinking length
# right y axis: language compliance rate (%)
# thinking length: bar chart
# language compliance rate: line chart (one line only for multi setting)


import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

# Set font and size
fp = fm.FontProperties(fname='assets/Times New Roman.ttf')
plt.rcParams['font.size'] = 12
# plt.rcParams['font.family'] = fp.get_name()

# Data preparation
datasets = ['math-500', 'aime2025', 'olymmath']
settings = ['en', 'multi', 'self']

accuracy = {
    'en': [79.35, 18.61, 6.58],
    'multi': [75.67, 16.67, 5],
    'self': [78.12, 18.89, 7]
}

pass_k = {
    'en': [91.6, 33.33, 19],
    'multi': [93.4, 30, 25],
    'self': [90, 30, 22]
}

thinking_length = {
    'en': [2900.67, 4056.24, 4084.87],
    'multi': [2339.74, 3966.70, 4065.73],
    'self': [2953.22, 4047.35, 4079.80]
}

compliance_rate = {
    'multi': [73.28, 71.94, 74.67]
}

# Create figure with two subplots vertically stacked
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 3.5))

x = np.arange(len(datasets))
width = 0.2

# Plot 1: Accuracy and Pass@k
ax1.bar(x - width, accuracy['en'], width, label='en', color='steelblue')
ax1.bar(x, accuracy['multi'], width, label='multi', color='orange')
ax1.bar(x + width, accuracy['self'], width, label='self', color='green')
ax1.set_ylabel('Accuracy (%)', fontproperties=fp)
ax1.tick_params(axis='y')

# Pass@k line on secondary axis
ax2 = ax1.twinx()
ax2.plot(x - width, pass_k['en'], '^-', label='en', color='steelblue', linewidth=2)
ax2.plot(x, pass_k['multi'], 'o-', label='multi', color='orange', linewidth=2)
ax2.plot(x + width, pass_k['self'], 'x-', label='self', color='green', linewidth=2)
ax2.set_ylabel('Pass@k (%)', fontproperties=fp)
ax2.tick_params(axis='y')

# Set x-axis labels
labels = ["MATH-500", "AIME2025", "OlymMath en-easy"]
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontproperties=fp)
ax1.set_xlabel('Dataset', fontproperties=fp)
ax1.set_title('Accuracy and Pass@k Comparison', fontproperties=fp)

# set y-axis limits
ax1.set_ylim(0, 100)
ax2.set_ylim(0, 100)

# Plot 2: Thinking Length and Language Compliance Rate
ax3.bar(x - width, thinking_length['en'], width, label='en', color='steelblue')
ax3.bar(x, thinking_length['multi'], width, label='multi', color='orange')
ax3.bar(x + width, thinking_length['self'], width, label='self', color='green')
ax3.set_ylabel('Thinking Length', fontproperties=fp)
ax3.tick_params(axis='y')

# Compliance rate line on secondary axis
ax4 = ax3.twinx()
ax4.plot(x, compliance_rate['multi'], 'o-', label='Language Compliance Rate', color='red', linewidth=2)
ax4.set_ylabel('Language Compliance Rate (%)', fontproperties=fp)
ax4.tick_params(axis='y')

# Set x-axis labels
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontproperties=fp)
ax3.set_xlabel('Dataset', fontproperties=fp)
ax3.set_title('Thinking Length and Language Compliance Rate', fontproperties=fp)

# Set y-axis limits
ax3.set_ylim(0, 4200)
ax4.set_ylim(0, 100)

fig.tight_layout()
plt.savefig('plot/results/combined_plots.png', dpi=300, bbox_inches='tight')
# plt.show()
