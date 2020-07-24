# General
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

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

# REad
folder = 'svm-none-median-smote-std-kfold-21'
folder = 'ann-none-median-smote-std-kfold-21'
folder = 'svm-none-median-smote-std-kfold-6'

data = pd.read_csv('./outputs/%s/results-micro_confirmed.csv' % folder)

# --------------------
# Load data
# --------------------
data['y'] = data['micro_confirmed']
data['y_pred'] = data['p1']>0.5
data['y_prob'] = data['p1']
data['type2'] = 'None'
data.loc[(data['y']==1) & (data['y_pred']==1), 'type2'] = 'tp'
data.loc[(data['y']==1) & (data['y_pred']==0), 'type2'] = 'fn'
data.loc[(data['y']==0) & (data['y_pred']==1), 'type2'] = 'fp'
data.loc[(data['y']==0) & (data['y_pred']==0), 'type2'] = 'tn'


"""
plt.figure()


sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="p1", y="type2", #hue="smoker",
               split=True, #inner="quart", #scale='count',
               whis=[0,10000],
               #palette={"Yes": "y", "No": "b"},
               data=data)
sns.despine(left=True)
"""

# --------------------------------------------
# Figure
# --------------------------------------------
# Create figure
plt.figure()

# Configure order
order = ['tn', 'fn', 'fp', 'tp']

# Counts
counts = data[['type2', 'p1']].groupby(by='type2').count()
ylabels = ['%s (%s)'%(c, counts.loc[c, 'p1']) for c in order]

# Show

# Configure seaborn.
sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Display (scale=width)
ax = sns.violinplot(x="p1", y="type2", hue="covid_confirmed",
                    order=order, split=True, scale='count', scale_hue=True,
                    whis=[0,10000], data=data,  saturation=0.8,
                    palette={1: "r", 0: "y"})

# Configure
ax.set_yticklabels(ylabels)
sns.despine(left=True)
plt.suptitle('Probability distributions')
plt.title('Folder: %s' % folder)
plt.xlabel('Probability of positive outcome')
plt.ylabel('')
#plt.tight_layout()

import time

# Save
plt.savefig('./outputs/_figures/crossed-covid-micro-distributions-%s.png' % time.strftime("%Y%m%d%H%M%S"))


# Show
plt.show()

import sys
sys.exit()
# -------------------
# Plot
# -------------------
# Feature, sites and target vectors
values = ['ALP', 'ALT', 'BIL', 'CRE', 'CRP', 'WBC', 'PCT', 'probs']
splits_covid = ['covid-n', 'covid-p']
splits_micro = ['micro-n', 'micro-p']

# Add covid positivity
data['covid-n'] = ~data['covid_confirmed'].astype(bool)
data['covid-p'] = data['covid_confirmed'].astype(bool)

# Add micro positivity
data['micro-n'] = ~data['micro_confirmed'].astype(bool)
data['micro-p'] = data['micro_confirmed'].astype(bool)




# -------------------
# Violin plot
# -------------------
# Draw a nested boxplot to show bills by day and time
"""
sns.boxplot( y='probs',
            hue='covid_confirmed', palette=["m", "g"],
            data=data)"""
plt.figure()
sns.violinplot(x='micro_confirmed', y="p1", data=data, hue='covid_confirmed',
            whis=[0, 100], palette="vlag")
sns.despine(offset=10, trim=True)

# Show
plt.tight_layout()
plt.show()



import sys
sys.exit()

print(data)

sns.set(style="whitegrid", palette="pastel", color_codes=True)


# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="type", y="p1", #hue="smoker",
               split=True, inner="quart",
               #palette={"Yes": "y", "No": "b"},
               data=data)
sns.despine(left=True)

plt.show()