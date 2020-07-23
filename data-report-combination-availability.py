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
n_samples = 10000

# Create frequency dataframe
frequency = pd.DataFrame()
frequency['count'] = data[biomarkers].count()
frequency = frequency[frequency['count']>n_samples]

# Define biomarkers variable
biomarkers_common = list(frequency.index.values)

biomarkers_common = ['cl', 'cre', 'crp']
# Show information
print("\nThere are <%s> biomarkers with more than <%s> samples: \n%s" % \
      (len(biomarkers_common), n_samples, biomarkers_common))


data = data[biomarkers_common]
print(data.isnull().groupby(biomarkers_common[:2]).size())
print(data.isnull().groupby(biomarkers_common).size())


b = sorted(['alb','alp','alt','baso','bil','cl','cre','crp','egfr',
                   'eos','hct','hgb','k','ly','mch','mchc','mcv','mono',
                   'mpv','neut','nrbca','plt','rbc','rdw','sodium','urea','wbc'])

for n in range(len(b)):
    # Create _combinations
    combinations = list(itertools.combinations(b, n))
    print("\nThere are <%s> combinations of length <%s>" % (len(combinations), n))

import sys
sys.exit()

# -----------------------------------
# Compute feature availability (Plot)
# -----------------------------------
# Create records list
records = []

for n in range(10, 15)[::-1]:

    # Create _combinations
    combinations = list(itertools.combinations(biomarkers_common, n))
    # Show information
    print("\nThere are <%s> combinations of length <%s>" % (len(combinations), n))

    for i, c in enumerate(combinations):
        if (i % 10000)==0:
            print("Processing... %s" % i)
        aux = data[list(c)].dropna(how='any')
        records.append({
            'combination': str(sorted(c)),
            'count': aux.shape[0],
            'length': aux.shape[1]
        })

    # Create dataframe from records list
    dataframe = pd.DataFrame(records)
    dataframe = dataframe.sort_values('count', ascending=False)
    dataframe.to_csv('./outputs/_combinations/combinations-%s.csv' % time.strftime("%Y%m%d%H%M%S"))

import sys
sys.exit()

# -----------------------------------
# Compute feature availability (Plot)
# -----------------------------------

# Create dataframe
# ----------------
dataframe = pd.DataFrame()
dataframe['count'] = data.count()
dataframe['count_covid19'] = data[data['covid_confirmed']].count()
dataframe['abbrev'] = dataframe.index
dataframe = dataframe.sort_values('count', ascending=False)
dataframe = dataframe.iloc[4:]

# Plot figure
# -----------
# Plot configuration
top = 60

# Create figure
f, ax = plt.subplots(figsize=(6, 10))

# Plot the total availability
sns.set_color_codes("pastel")
sns.barplot(x='count', y='abbrev', data=dataframe.head(top),
            label='Total', color='b')

# Plot the covid19 positive availability
sns.set_color_codes("muted")
sns.barplot(x='count_covid19', y='abbrev', data=dataframe.head(top),
            label="Covid19+", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=[0, n_profiles],
       ylabel="",
       xlabel="Presence of biomarker in daily profiles")
sns.despine(left=True, bottom=True)

plt.suptitle('Presence of biomarkers in daily profiles (Top 70)')
plt.title('#Profiles: %s | #Biomarkers: %s' % (len(data), len(data.columns[5:])))
plt.tight_layout()
plt.show()



#_combinations = itertools._combinations(data.columns[5:], 5)

#print(len(data))
#print(len(list(_combinations)))
