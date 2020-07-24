# Libraries
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------
# Configuration
# ---------------------
sns.set(style="whitegrid")

# Automatic save
auto_save = False

# Define common top
top = 60

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


# ------------------------------------
# Create dataframe and plot
# ------------------------------------
# Create dataframe for plotting
dataframe = pd.DataFrame()
dataframe['count_total'] = data.count()
dataframe['count_covid'] = data[data['covid_confirmed']].count()
dataframe['count_micro'] = data[data['micro_confirmed']].count()
dataframe = dataframe.sort_values('count_total', ascending=False)

# Clean
dataframe = dataframe.iloc[1:]
dataframe = dataframe.rename(index={'covid_confirmed':'Total'})
dataframe.reset_index(inplace=True)

# Show information
print(dataframe)

# Plot
# ----
# Create figure
f, ax = plt.subplots(figsize=(6, 5))

# Plot the total availability
sns.set_color_codes("pastel")
sns.barplot(x='count_total', y='index',
            data=dataframe.head(top),
            label='Total', color='b')

# Plot the covid19 positive availability
sns.set_color_codes("muted")
sns.barplot(x='count_covid', y='index',
            data=dataframe.head(top),
            label="Covid+", color='b')

# Plot the micro positive availability
sns.set_color_codes("muted")
sns.barplot(x='count_micro', y='index',
            data=dataframe.head(top),
            label="Micro+", color='r')

# Configure axes
ax.legend(ncol=1, loc="lower right", frameon=True)
ax.set(xlim=[0, n_profiles], ylabel="",
       xlabel="Presence of biomarker in daily profiles")
sns.despine(left=True, bottom=True)

# Configure
plt.suptitle('Presence of biomarkers in daily profiles (Top 70)')
plt.title('#Profiles: %s | #Biomarkers: %s' % \
          (len(data), len(data.columns[5:])))
plt.tight_layout()

# Save
if auto_save:
    plt.savefig('./outputs/_figures/biomarker-availability-%s.eps' % time.strftime("%Y%m%d%H%M%S"))

# Display
plt.show()
