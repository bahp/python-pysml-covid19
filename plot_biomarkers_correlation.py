# Libraries
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------
# Configuration
# ---------------------
sns.set(style="whitegrid")

# Automatic save
auto_save = False

# Consider those biomarkers with are present in
# more than n_sample profiles. To see this number
# execute plot_biomarkers_availability.py
n_samples = 8000

# This columns should be removed from the loaded dataset
# to respect the requirements of the plotting code as
# stated below.
drop_columns = ['_uid', 'date_result']

# -----------------
# Read data
# -----------------
# Read
data = pd.read_csv('./data/covid19/daily_profiles.csv')
# Columns to lowercase
data.columns = [c.lower() for c in data.columns]
# Drop columns
data = data.drop(columns=drop_columns)


# ------------------------------------
# Print brief summary
# ------------------------------------
# General variables
biomarkers = list(data.columns[:-2])
n_profiles = len(data)
n_biomarkers = (len(biomarkers))

# Show information
print("\n{0}\n{1}\n{2}".format("-" * 40, "Report", "-" * 40))
print("Number of rows: %s" % n_profiles)
print("Number of biomarkers: %s" % n_biomarkers)
print("List of biomarkers:\n%s" % data.columns.values)


# --------------------------
# Get most common biomarkers
# --------------------------
# Create frequency dataframe
frequency = pd.DataFrame()
frequency['count'] = data[biomarkers].count()
frequency = frequency[frequency['count']>n_samples]

# Define biomarkers variable
biomarkers_common = frequency.index.values

# Show information
print("\nThere are <%s> biomarkers with more than <%s> samples: \n%s" % \
      (len(biomarkers_common), n_samples, biomarkers_common))

# -----------------------------------
# Compute feature correlation (Plot)
# -----------------------------------
# Compute correlation
corr = data[biomarkers_common].corr()*100

# Reorder for display purposes
corr = corr.reindex(sorted(corr.columns), axis=1)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)


# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, #vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            annot=True, fmt=".0f", annot_kws={"size": 10})

# Configure
plt.suptitle('Correlation of biomarkers in daily profiles (>%s samples)' % n_samples)
plt.title('#Profiles: %s | #Biomarkers: %s' % (len(data), len(data.columns[5:])))

# Save
if auto_save:
    plt.savefig('./outputs/_figures/biomarker-correlation-%s.png' % time.strftime("%Y%m%d%H%M%S"))


# Show
plt.show()