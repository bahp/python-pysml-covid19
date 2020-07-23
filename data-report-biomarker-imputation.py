# Libraries
import time
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Imputer
from sklearn.experimental import enable_iterative_imputer

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline


# ----------------
# Constants
# ----------------
rng = np.random.RandomState(42)


# ---------------------------------------------
# helper methods
# ---------------------------------------------
def add_missing_values(X_full):
    """

    :param X_full:
    :param y_full:
    :return:
    """
    n_samples, n_features = X_full.shape

    # Add missing values in 75% of the lines
    missing_rate = 0.70
    n_missing_samples = int(n_samples * missing_rate)

    missing_samples = np.zeros(n_samples, dtype=np.bool)
    missing_samples[: n_missing_samples] = True

    rng.shuffle(missing_samples)
    missing_features = rng.randint(0, n_features, n_missing_samples)
    X_missing = X_full.copy()
    X_missing[missing_samples, missing_features] = np.nan

    return X_missing


# ---------------------
# Configuration
# ---------------------
sns.set(style="whitegrid")

# -----------------
# Read data
# -----------------
# Load pandas dataframe
data = pd.read_csv('./data/covid19/one_line_values_pathology.csv')
# Columns to lowercase
data.columns = [c.lower() for c in data.columns]

# General variables
biomarkers = data.columns[5:]
n_profiles = len(data)
n_biomarkers = (len(biomarkers))

# Show information
print("\n{0}\n{1}\n{2}".format("-" * 40, "Report", "-" * 40))
print("Number of rows: %s" % n_profiles)
print("Number of biomarkers: %s" % n_biomarkers)

# -----------------------------------
# Prepare dataset
# -----------------------------------
# Libraries
from pyPI.preprocessing.filters import IQRFilter

# Define features
features = ['alp', 'alt', 'bil', 'cre', 'crp', 'wbc']

# Filter data by features
data = data[features]

# Remove outliers
data = IQRFilter().fit_filter(X=data)

# Keep only full profiles
data = pd.DataFrame(data, columns=features).dropna(how='any')

# Input missing data
data_missing = add_missing_values(data.to_numpy())

# Describe data
print(data.describe())

# ------------------
# Define imputers
# ------------------
# Libraries
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

# List of imputers
imputers = [('Simple-Mean', SimpleImputer(strategy='mean')),
            ('Simple-Median', SimpleImputer(strategy='median')),
            ('KNN', KNNImputer()),
            ('Iterative-Ridge', IterativeImputer(random_state=0, estimator=BayesianRidge())),
            ('Iterative-DTree', IterativeImputer(random_state=0, estimator=DecisionTreeRegressor())),
            ('Iterative-ETree', IterativeImputer(random_state=0, estimator=ExtraTreesRegressor(n_estimators=10))),
            ('Iterative-KNReg', IterativeImputer(random_state=0, estimator=KNeighborsRegressor(n_neighbors=10)))
]


# ---------------------
# Loop through imputers
# ---------------------
# Libraries
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Create records
records = []

# Loop
for name, imp in imputers:
    # Fit imputer
    imp.fit(data_missing)
    # Transform missing data
    data_imputed = imp.transform(data_missing)

    for i,c in enumerate(data.columns):
        idxs = np.isnan(data_missing[:,i])
        y_true = data.to_numpy()[:, i]
        y_pred = data_imputed[:, i]

        records.append({
            'biomarker': c,
            'method': str(name),
            'mse': mean_squared_error(y_true[idxs], y_pred[idxs]),
            'mae': mean_absolute_error(y_true[idxs], y_pred[idxs])
        })



# Create scores dataframe
df = pd.DataFrame(records)

# Plot figure
# -----------
# Create figure
f, ax = plt.subplots(2, 1, figsize=(10, 6))
axes = ax.flatten()

# Display
sns.set_color_codes("pastel")
sns.barplot(x='method', y='mae', hue='biomarker', data=df, ax=axes[0])



# Create title
title = '#Profiles: %s\n' % data.shape[0]
title += '%s' % pd.DataFrame(data_missing, columns=features).isnull().sum().to_frame().T

# Title
axes[0].set_title(title)
axes[1].text(x=0.01, y=0.01, s=str(data.describe()), fontdict={'family':'monospace'}, clip_on=True)
axes[1].set_xticks([])
axes[1].set_yticks([])

plt.xticks(rotation=45)
plt.suptitle("Comparison of imputation methods")
plt.tight_layout()

# Save
plt.savefig('./outputs/_figures/biomarker-imputation-%s.jpg' % time.strftime("%Y%m%d%H%M%S"))

# Show
plt.show()



