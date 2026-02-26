import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

# 1. Load the data
def extract_dataset(filename):
    """Extracts a dataset from a CSV file."""
    return pd.read_csv(filename)

datasets = {
    'Math-500': extract_dataset("plot/results/language_rates/math-500.csv"),
    'AIME-2025': extract_dataset("plot/results/language_rates/aime25.csv"),
    'OlymMath en-easy': extract_dataset("plot/results/language_rates/olymmath.csv")
}

# 2. Accuracy Data Mapping
accuracies = {
    'Math-500': [0.7812, 0.7203, 0.8297, 0.9145],
    'AIME-2025': [0.1889, 0.1611, 0.2028, 0.3194],
    'OlymMath en-easy': [0.07, 0.0542, 0.0967, 0.2417]
}

# 3. Setup Plotting
languages = datasets['Math-500'].columns[1:]
stages = datasets['Math-500']['stage'].tolist()
colors = plt.cm.tab20(np.linspace(0, 1, len(languages)))
fp = fm.FontProperties(fname='assets/Times New Roman.ttf')
plt.rcParams['font.size'] = 16
fig, axes = plt.subplots(3, 1, figsize=(5, 7.5), sharex=True)
for ax in axes:
    ax.tick_params(axis='x', labelbottom=True)

for i, (name, d) in enumerate(datasets.items()):
    ax = axes[i]
    
    # --- Stackplot (Primary Y-axis) ---
    y_values = [d[lang].values for lang in languages]
    ax.stackplot(stages, y_values, labels=languages, colors=colors, alpha=0.7)
    
    ax.set_title(f'Language Selection Rates: {name}', fontproperties=fp, fontweight='bold')
    ax.set_ylabel('Selection Rate (%)', fontproperties=fp)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # --- Accuracy Line (Secondary Y-axis) ---
    ax2 = ax.twinx()  # Create the double y-axis
    acc_line = ax2.plot(stages, accuracies[name], color='grey', marker='o', linestyle='dotted', 
                        linewidth=2, markersize=3, label='Accuracy')
    
    ax2.set_ylabel('Accuracy', fontproperties=fp, color='black')
    # ax2.set_ylim(0, 1.0) # Set to 1.0 since these look like ratios
    
    # Formatting ticks
    for a in [ax, ax2]:
        a.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        for tick in a.get_xticklabels() + a.get_yticklabels():
            tick.set_fontproperties(fp)

    # --- Legend Management ---
    if i == 2:
        # Combine handles from stackplot and the line plot
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        
        fig.legend(
            handles1 + handles2, labels1 + labels2,
            loc='lower center',
            ncol=5,
            prop=fp,
            frameon=True,
            fontsize='small'
        )

plt.xlabel('Training Stages', fontproperties=fp)
plt.tight_layout(rect=[0, 0.1, 1, 1])

# Save and show
plt.savefig('plot/results/language_selection_trends.pdf', dpi=300)
# plt.show()