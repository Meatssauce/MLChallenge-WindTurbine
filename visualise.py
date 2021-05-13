from tools import *

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = read_csv_and_drop_invalid(TRAIN_FILE_NAME, TARGET_FEATURE)
X, y = df.drop(columns=[TARGET_FEATURE]), df[TARGET_FEATURE].copy()

X = preprocessor.fit_transform(X, y)

###############################################################################
#                3a. Exploratory Data Analysis: Correlation Matrix            #
###############################################################################
# Make correlation matrix
corr_matrix = X.corr(method = 'spearman').abs()

# Set font scale
sns.set(font_scale = 1)

# Set the figure size
f, ax = plt.subplots(figsize=(12, 12))

# Make heatmap
sns.heatmap(corr_matrix, cmap= 'YlGnBu', square=True, ax = ax)

# Tight layout
f.tight_layout()

# Save figure
f.savefig('correlation_matrix.png', dpi = 1080)

###############################################################################
#                  3b. Exploratory Data Analysis: Box Plots                   #
###############################################################################
# Set graph style
sns.set(font_scale = 0.75)
sns.set_style({'axes.facecolor': '1.0', 'axes.edgecolor': '0.85', 'grid.color': '0.85',
               'grid.linestyle': '-', 'axes.labelcolor': '0.4', 'xtick.color': '0.4',
               'ytick.color': '0.4', 'axes.grid': False})

# Create box plots based on feature type

# Set the figure size
f, ax = plt.subplots(figsize=(9, 12))
sns.boxplot(data=X, orient="h", palette="Set2")

# Set axis label
plt.xlabel('Feature Value')

# Tight layout
f.tight_layout()

# Save figure
f.savefig(f'Box Plots.png', dpi = 1080)