# Libraries
import itertools
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------
# Configuration
# ---------------------
sns.set(style="whitegrid")

# --------------------------------------------------------------
# Read data
# --------------------------------------------------------------
# Load pandas dataframe
data = pd.read_csv('./data/covid19/one_line_values_pathology.csv')
# Columns to lowercase
data.columns = [c.lower() for c in data.columns]

# General variables
biomarkers = list(data.columns[5:])
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
biomarkers_common = list(frequency.index.values)

# Show information
print("\nThere are <%s> biomarkers with more than <%s> samples: \n%s" % \
      (len(biomarkers_common), n_samples, biomarkers_common))

# -----------------------------------
# Compute combinations (Plot)
# -----------------------------------
# Select data
biomarkers_panels = sorted(biomarkers_common)
biomarkers_panels = ['alp', 'alt', 'bil', 'cre', 'crp', 'wbc']

# Create dataframe
panels = data.notnull().groupby(biomarkers_panels).size()
panels = panels.sort_values(ascending=False)
panels.to_csv('./outputs/_combinations/combinations-%s.csv' % time.strftime("%Y%m%d%H%M%S"))

# Show
print(panels)
print("Total sum: %s" % panels.sum())