import numpy as np
from scipy.stats import pearsonr, spearmanr
import pandas as pd

# Load accruracy data for each dataset
accuracy_data = pd.read_csv("plot/results/lang_acc.csv")

# Load language selection rates for each dataset
language_rates = {
    'math-500': pd.read_csv("plot/results/language_rates/math-500.csv"),
    'aime25': pd.read_csv("plot/results/language_rates/aime25.csv"),
    'olymmath': pd.read_csv("plot/results/language_rates/olymmath.csv")
}

# breakpoint()

# Calculate correlations for "sft" and "explore" for each dataset, with p values
correlation_results = {}
for dataset in accuracy_data['dataset']:
    accs = accuracy_data.loc[accuracy_data['dataset'] == dataset].iloc[:, 1:].values.flatten()
    sft_rates = language_rates[dataset].loc[language_rates[dataset]['stage'] == 'sft'].iloc[:, 1:].values.flatten()
    explore_rates = language_rates[dataset].loc[language_rates[dataset]['stage'] == 'explore'].iloc[:, 1:].values.flatten()
    
    # Calculate Pearson and Spearman correlations for "sft"
    pearson_sft, p_pearson_sft = pearsonr(accs, sft_rates)
    spearman_sft, p_spearman_sft = spearmanr(accs, sft_rates)
    
    # Calculate Pearson and Spearman correlations for "explore"
    pearson_explore, p_pearson_explore = pearsonr(accs, explore_rates)
    spearman_explore, p_spearman_explore = spearmanr(accs, explore_rates)
    
    correlation_results[dataset] = {
        'sft': {
            'pearson': (pearson_sft, p_pearson_sft),
            'spearman': (spearman_sft, p_spearman_sft)
        },
        'explore': {
            'pearson': (pearson_explore, p_pearson_explore),
            'spearman': (spearman_explore, p_spearman_explore)
        }
    }

# Print results
for dataset, results in correlation_results.items():
    print(f"Dataset: {dataset}")
    print(f"  SFT - Pearson: {results['sft']['pearson'][0]:.4f} (p={results['sft']['pearson'][1]:.4f}), Spearman: {results['sft']['spearman'][0]:.4f} (p={results['sft']['spearman'][1]:.4f})")
    print(f"  Explore - Pearson: {results['explore']['pearson'][0]:.4f} (p={results['explore']['pearson'][1]:.4f}), Spearman: {results['explore']['spearman'][0]:.4f} (p={results['explore']['spearman'][1]:.4f})")