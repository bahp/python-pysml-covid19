# General
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# Load data
# ------------------------------
# Load
patho = pd.read_csv('./outputs/_datasets/daily_profiles_pathology.csv',
                    parse_dates=['date_result'],
                    dtype={'nhsnumber': str})
micro = pd.read_csv('./outputs/_datasets/daily_profiles_microbiology.csv',
                    parse_dates=['ReceiveDate', 'FinalDate'],
                    dtype={'caseNHSNumber': str})

# Format columns
patho.columns = [c.lower() for c in patho.columns]

# Filter by date
patho = patho[patho.date_result > '2020-06-01']
micro = micro[micro.ReceiveDate > '2020-06-01']

# Add count days in pathology
patho['date_init'] = patho.groupby('nhsnumber').date_result.transform('min')
patho['date_diff'] = (patho['date_result'] - patho['date_init']) / np.timedelta64(1, 'D')

# Show information
print("Total NHS Numbers: %s" % patho.nhsnumber.nunique())

# -------------
# Manual plot
# -------------
# Create figure
fig, ax = plt.subplots(6, 7, figsize=(15,10))
axes = ax.flatten()

# Loop for each patient
for i,nhsnumber in enumerate(patho.nhsnumber.unique()):

    # Show outcome
    micro_confirmed = nhsnumber in micro.caseNHSNumber.unique()

    # Get dataframe
    df = patho[patho.nhsnumber==nhsnumber]
    df = df[df._nan<=0]
    df = df.set_index('date_result')
    df.plot(y='micro_p1', ax=axes[i], marker='o',
            ylim=[0,1.1], legend=False, linewidth=1)

    print("\n\n")
    print(nhsnumber)
    # Same record repeated several times review!
    print(df[['nhsnumber', 'alp', 'alt', 'bil', 'cre', 'crp', 'wbc']].drop_duplicates())

    # Horizontal line
    axes[i].axhline(y=0.5, ls = ":", c = ".5")

    # Get datframe
    df = micro[micro.caseNHSNumber==nhsnumber]
    date_received = df.ReceiveDate.values
    date_reported = df.FinalDate.values

    for e in date_received:
        if not pd.isnull(e):
            axes[i].axvline(e, color='k', linestyle='--')

    for e in date_reported:
        if not pd.isnull(e):
            axes[i].axvline(e, color='r', linestyle='--')

    axes[i].set_title("{0}".format(nhsnumber))

    # Set x ticks
    #axes[i].set_xticks(np.arange(len(axes[i].get_xticks())))
    axes[i].set_xticklabels([])
    axes[i].set_yticks([0,1])



# Final configuration
plt.subplots_adjust(left=0.05, right=0.95, wspace=0.5)
plt.tight_layout()
plt.show()

import sys

sys.exit()

# Merge
merged = patho.merge(micro, how='outer', left_on='nhsnumber', right_on='caseNHSNumber')

# Add information
merged['date_init'] = merged.groupby('nhsnumber').date_result.transform('min')
merged['date_diff'] = (merged['date_result'] - merged['date_init']) / np.timedelta64(1, 'D')

merged = merged.set_index(['nhsnumber', 'date_diff'])
print(merged)
import sys

sys.exit()
data['date_result'] = pd.to_datetime(micro.date_result)

micro = pd.read_csv('./outputs/_datasets/micro_dates.csv',
                    dtype={'caseNHSNumber': str})
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
