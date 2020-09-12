# General
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------
# Constants
# -------------------
# Default features for prediction
_DEFAULT_FEATURES = {
    '6': sorted(['alp', 'alt', 'bil', 'cre', 'crp', 'wbc']),

    '7': sorted(['alp', 'alt', 'bil', 'cre', 'crp', 'wbc', 'pct']),

    '21': sorted(['alb', 'alp', 'alt', 'baso', 'bil', 'cl', 'cre',
                  'crp', 'egfr', 'eos', 'k', 'ly', 'mcv', 'mono',
                  'mpv', 'nrbca', 'plt', 'rbc', 'rdw', 'urea', 'wbc']),

    '27': sorted(['alb', 'alp', 'alt', 'baso', 'bil', 'cl', 'cre', 'crp',
                  'egfr', 'eos', 'hct', 'hgb', 'k', 'ly', 'mch', 'mchc',
                  'mcv', 'mono', 'mpv', 'neut', 'nrbca', 'plt', 'rbc',
                  'rdw', 'sodium', 'urea', 'wbc'])
}

# Define feature group
slug = '21'

# Feature vector
features = _DEFAULT_FEATURES[slug]

# Targets
target = 'micro_confirmed'
compare = 'covid_confirmed'

# Folder name
outpath = './outputs'

# Pipeline
pipelinepath = '%s/inference-compare/ann-none-median-smote-std-kfold-21' % outpath

# ------------------------------
# Load estimators
# ------------------------------
# Pipeline micro
pipeline_micro = '%s/inference-compare/ann-none-median-smote-std-kfold-21' % outpath
estimator_micro = \
    pickle.load(open('%s/estimators/iteration_00/estimator_00.pkl' % pipeline_micro, 'rb'))

# Pipeline covid
# SVM - 3
# ANN - 26
pipeline_covid = '%s/inference-covid/ann-none-median-smote-std-skfold-21' % outpath
estimator_covid = \
    pickle.load(open('%s/estimators/iteration_00/estimator_26.pkl' % pipeline_covid, 'rb'))

# ------------------------------
# Load data
# ------------------------------
data = pd.read_csv('./data/ipc-sarscov-inpatients-sitrep-20200806/daily_profiles.csv')
data.columns = [c.lower() for c in data.columns]
# data = data[features + [target, compare]]
# data.covid_confirmed = data.covid_confirmed.astype(int)
# data.micro_confirmed = data.micro_confirmed.astype(int)

print(features)

# Add probabilities
data['covid_p1'] = estimator_covid.predict_proba(data[features].to_numpy())[:, 1]
data['micro_p1'] = estimator_micro.predict_proba(data[features].to_numpy())[:, 1]
data['date_result'] = pd.to_datetime(data.date_result)

print(data.columns.values)
print(data.date_result)
print(data.dtypes)



# Add info
data = data[data.date_result > '2020-07-01']
data['date_init'] = data.groupby('nhsnumber').date_result.transform('min')
data['date_diff'] = (data['date_result'] - data['date_init']) / np.timedelta64(1, 'D')


# Load dates
print(data[['nhsnumber', 'date_init', 'date_diff']])

micro = pd.read_csv('./outputs/_datasets/micro_dates.csv',
                    dtype={'caseNHSNumber':str})
micro = micro[['PtNumber', 'caseNHSNumber', 'ReceiveDate']].drop_duplicates()

"""
#micro = micro.merge(data, how='left', left_on='caseNHSNumber', right_on='nhsnumber')
data = data.merge(micro, how='outer', left_on='nhsnumber', right_on='caseNHSNumber')
print(data[['nhsnumber', 'date_init', 'date_diff', 'ReceiveDate']])
#print(micro[['nhsnumber', 'date_init', 'date_diff', 'ReceiveDate']])
#print(micro.caseNHSNumber.unique())
"""

print(data[['nhsnumber', 'date_init']].drop_duplicates())
print(micro)




# --------------------------
# Plot covid predictions
# --------------------------
# Plot only full
data_plot = data[features + ['nhsnumber',
                             'date_result',
                             'micro_p1',
                             'covid_p1',
                             'date_diff',
                             'micro_confirmed']]
data_plot = data_plot.dropna(how='any')




# --------------------
# Seaborn
# --------------------
# Constants
x_max = data_plot.date_diff.max()

# Settings
sns.set(style="ticks")

# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(data_plot, col="nhsnumber",
                     hue="nhsnumber", palette="tab20c",
                     col_wrap=6, height=1.5, sharex=False)

# Draw a horizontal line to show the starting point
grid.map(plt.axhline, y=0.5, ls=":", c=".5")

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "date_diff", "micro_p1", marker="o")

# Adjust the tick positions and labels
grid.set(ylim=(0, 1.1))

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=0.5)

plt.subplots_adjust(left=0.05, right=0.95, wspace=0.5)
plt.show()

import sys
sys.exit()




fig, ax = plt.subplots(5, 8, figsize=(20, 6))
axes = ax.flatten()

for i, (nhsnuber, p) in enumerate(data_plot.groupby(by='nhsnumber')):
    p[['date_diff', 'covid_p1']].set_index('date_diff').plot(ax=axes[i],
                                                             legend=False, ylim=[0, 1.1], marker='o',
                                                             markersize=2.5, linewidth=1.5)

plt.subplots_adjust(left=0.05, right=0.95, hspace=1.5)

# sns.lineplot(x='date_result', y='covid_p1', hue='nhsnumber')
plt.axis('off')
plt.show()

import sys

sys.exit()

# Save
for p in data.groupby(by='nhsnumber'):
    plt.figure()

    import sys

    sys.exit()
