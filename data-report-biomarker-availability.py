# Libraries
import time
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
   .. todo: Generalis method and just format reading the dataframe.
"""

# ---------------------
# Configuration
# ---------------------
sns.set(style="whitegrid")

color = 'r'
label = 'micro_confirmed'

# -----------------
# Read data
# -----------------
# Load pandas dataframe
#data = pd.read_csv('./data/covid19/one_line_values_pathology.csv')
data = pd.read_csv('./data/covid19/daily_profiles.csv')
# Columns to lowercase
data.columns = [c.lower() for c in data.columns]

# General variables
biomarkers = ['micro_confirmed'] + list(data.columns[2:-2])
#biomarkers = list(data.columns[5:])
n_profiles = len(data)
n_biomarkers = (len(biomarkers))

# Show information
print("\n{0}\n{1}\n{2}".format("-" * 40, "Report", "-" * 40))
print("Number of rows: %s" % n_profiles)
print("Number of biomarkers: %s" % n_biomarkers)

# -----------------------------------
# Compute feature availability (Plot)
# -----------------------------------

# Create dataframe
# ----------------
dataframe = pd.DataFrame()
dataframe['count'] = data.count()
dataframe['count_covid19'] = data[data[label]].count()
dataframe['abbrev'] = dataframe.index
dataframe = dataframe.sort_values('count', ascending=False)
dataframe = dataframe.iloc[3:]
#dataframe = dataframe.iloc[4:]

# Plot figure
# -----------
# Plot configuration
top = 60

# Create figure
f, ax = plt.subplots(figsize=(6, 10))

# Plot the total availability
sns.set_color_codes("pastel")
sns.barplot(x='count', y='abbrev', data=dataframe.head(top),
            label='Total', color=color)

# Plot the covid19 positive availability
sns.set_color_codes("muted")
sns.barplot(x='count_covid19', y='abbrev', data=dataframe.head(top),
            label="Micro+", color=color)

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=[0, n_profiles],
       ylabel="",
       xlabel="Presence of biomarker in daily profiles")
sns.despine(left=True, bottom=True)

# Configure
plt.suptitle('Presence of biomarkers in daily profiles (Top 70)')
plt.title('#Profiles: %s | #Biomarkers: %s' % (len(data), len(data.columns[5:])))
plt.tight_layout()

# Save
plt.savefig('./outputs/_figures/biomarker-availability-%s.png' % time.strftime("%Y%m%d%H%M%S"))

# Display
plt.show()
