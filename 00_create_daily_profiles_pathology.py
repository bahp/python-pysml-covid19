# Libraries
import pandas as pd
import numpy as np
import pickle

# ---------------------
# Constants
# ---------------------
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


# Filepaths
folder = './data/ipc-sarscov-inpatients-sitrep-20200806'
filepath_patho = '%s/sql029-pathology.csv' % folder
filepath_pims = '%s/pims-patients.csv' % folder
filepath_micro = './outputs/_datasets/micro_dates.csv'

# -----------------------
# Load pathology raw data
# -----------------------
# Load data
patho = pd.read_csv(filepath_patho, parse_dates=['dateResult'])

# Fix sodium (generalise for all orderCode with null value)
patho.loc[patho['orderCodeText']=='Sodium', 'orderCode'] = 'Sodium'

# Fix add date_result
patho['date_result'] = patho['dateResult'].dt.date

# Pivote
pivoted = pd.pivot_table(patho, values='result',
                                index=['NHSNumber', 'date_result'],
                                columns=['orderCode'],
                                aggfunc=np.mean)
pivoted.reset_index(inplace=True)

# ----------------
# Merge with PIMS
# ----------------
# Load pims
pims = pd.read_csv(filepath_pims)

# Merge with pims
pivoted = pivoted.merge(pims, how='left',
                              left_on='NHSNumber',
                              right_on='caseNHSNumber')

# Lower
pivoted.columns = [c.lower() for c in pivoted.columns]

# -----------------
# Add predictions
# -----------------
# Folder name
outpath = './outputs'

# Pipeline
pipelinepath = '%s/inference-micro/ann-none-median-smote-std-skfold-21' % outpath

# ------------------------------
# Load estimators
# ------------------------------
# Pipeline micro
pipeline_micro = '%s/inference-micro/svm-none-median-smote-std-skfold-21' % outpath
estimator_micro = \
    pickle.load(open('%s/estimators/iteration_00/estimator_03.pkl' % pipeline_micro, 'rb'))



# Add probabilities
pivoted['micro_p1'] = estimator_micro.predict_proba( \
    pivoted[_DEFAULT_FEATURES['21']].to_numpy())[:, 1]
pivoted['_nan'] = pivoted[_DEFAULT_FEATURES['21']].isnull().sum(axis=1)




"""
# ------------------
# Add micro outcome
# ------------------
# Read microbiology
micro = pd.read_csv(filepath_micro)

# Add microbiology confirmation
pivoted['micro_confirmed'] = False
pivoted.loc[pivoted.caseHospitalId.isin(micro.PtNumber.unique()), 'micro_confirmed'] = True

# ------------------
# Add covid outcome
# ------------------
pivoted['covid_confirmed'] = True
"""
# -------------------
# Count number of NaN
# -------------------
#pivoted['_nan'] = pivoted.isnull().sum(axis=1)

# Save
pivoted.to_csv('./outputs/_datasets/daily_profiles_pathology.csv', index=False)


import sys
sys.exit()
