# General
import time
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Note that we have develop algorithms to predict culture positivity. The aim of this
# graph is to evaluate whether the presence of covid affects the probabilities
# provided by this algorithms. If no effect then the distributions for covid+/-
# should be very similar.

# -------------------
# Configuration
# -------------------
# Ignore all the warnings
warnings.simplefilter('ignore')

# Set sns configuration
sns.set(style="ticks", palette="pastel")

# Set pandas configuration.
pd.set_option('display.max_colwidth', 14)
pd.set_option('display.width', 150)
pd.set_option('display.precision', 4)

# Save
auto_save = False

# Select the algorithm
folder = 'svm-none-median-smote-std-kfold-21'
folder = 'ann-none-median-smote-std-kfold-21'
#folder = 'svm-none-median-smote-std-kfold-6'
folder = 'svm-none-median-smote-std-kfold-21'

# Read data
data = pd.read_csv('./outputs/inference-compare/%s/results-micro_confirmed.csv' % folder)

# --------------------
# Load data
# --------------------
# Create the data for plotting
dataframe = pd.DataFrame()
dataframe['micro_confirmed'] = data['micro_confirmed']
dataframe['covid_confirmed'] = data['covid_confirmed']
dataframe['y'] = data['micro_confirmed']
dataframe['y_pred'] = data['p1']>0.5
dataframe['y_prob'] = data['p1']
dataframe['type'] = 'None'
dataframe.loc[(dataframe['y']==1) & (dataframe['y_pred']==1), 'type'] = 'tp'
dataframe.loc[(dataframe['y']==1) & (dataframe['y_pred']==0), 'type'] = 'fn'
dataframe.loc[(dataframe['y']==0) & (dataframe['y_pred']==1), 'type'] = 'fp'
dataframe.loc[(dataframe['y']==0) & (dataframe['y_pred']==0), 'type'] = 'tn'

# Counts for the y labels
counts = dataframe[['type', 'y_prob']].groupby(by='type').count()

# --------------------------------------------
# Figure
# --------------------------------------------
# Create figure
plt.figure()

# Configure order
order = ['tn', 'fn', 'fp', 'tp']

# The ylabels
ylabels = ['%s (%s)'%(c, counts.loc[c, 'y_prob']) for c in order]

# Configure seaborn.
sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Display (scale=width)
ax = sns.violinplot(x="y_prob", y="type", hue="covid_confirmed",
                    order=order, split=True, scale='count', scale_hue=True,
                    whis=[0,10000], data=dataframe,  saturation=0.8,
                    palette={1: "r", 0: "y"})

# Configure
ax.set_yticklabels(ylabels)
sns.despine(left=True)
plt.suptitle('Probability distributions')
plt.title('Folder: %s' % folder)
plt.xlabel('Probability of positive outcome')
plt.ylabel('')
#plt.tight_layout()

# Save
if auto_save:
    # Save
    plt.savefig('./outputs/_figures/crossed-covid-micro-distributions-%s.png' % time.strftime("%Y%m%d%H%M%S"))

# Show
plt.show()