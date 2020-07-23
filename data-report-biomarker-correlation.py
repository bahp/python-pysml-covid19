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

# ---------------
# Read data
# ---------------
# Load pandas dataframe
data = pd.read_csv('./data/covid19/one_line_values_pathology.csv')
# Columns to lowercase
data.columns = [c.lower() for c in data.columns]

# General variables
biomarkers = data.columns[5:]
n_profiles = len(data)
n_biomarkers = (len(biomarkers))

# Show information
print("\n{0}\n{1}\n{2}".format("-"*40, "Report", "-"*40))
print("Number of rows: %s" % n_profiles)
print("Number of biomarkers: %s" % n_biomarkers)

# --------------------------
# Get most common biomarkers
# --------------------------
# Configuration
n_samples = 8000

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
plt.savefig('./outputs/_figures/biomarker-correlation-%s.jpg' % time.strftime("%Y%m%d%H%M%S"))


# Show
plt.show()