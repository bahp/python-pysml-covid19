# Date: 14/06/2018
# Filename:
# License:
#
# TO DO: The grid search function is generic and very slow since it creates
#        a CBR instance for each iteration; that is, for each k. This causes
#        that the tree is created k times for each metric. However, it is
#        only necessary to create the tree once and then change the value
#        of n_neighbors in prediction.
#
# Description: In this example the performance of different classifiers is
#              compared.
#
#
###############################################################################
# Division
from __future__ import division

# Generic libraries
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

# Specific
from sklearn.model_selection import train_test_split

"""
# Estimator filepath
path_micro = './outputs/inference-micro/svm-none-median-smote-std-skfold-21/estimators/iteration_00/estimator_00.pkl'
path_covid = './outputs/inference-covid/svm-none-median-smote-std-skfold-21/estimators/iteration_00/estimator_00.pkl'

# Load estimators
estm_micro = pickle.load(open(path_micro, 'rb'))
estm_covid = pickle.load(open(path_covid, 'rb'))

it0_path = './outputs/inference-micro/svm-none-median-smote-std-skfold-21/iterations/iteration_00.pkl'

hos = np.load('./outputs/inference-micro/svm-none-median-smote-std-skfold-21/data/HOS.npy')
"""

from sklearn.model_selection import train_test_split

# Imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Import own libraries
from pyPI.preprocessing.filters import IQRFilter
from pyPI.pipeline.pipeline import Pipeline

# Ignore all the warnings
warnings.simplefilter('ignore')

# -----------------------------------------------------------------------------
#                              CONFIGURATION
# -----------------------------------------------------------------------------
# ------------------------------
# algorithm constants
# ------------------------------
# Gaussian naive bayes (GNB)
gnb_grid = {'priors': [None]}

# Logistic regression grid (LLG)
log_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1.0],
    'fit_intercept': [True, False],
    'class_weight': [None, 'balanced'],
    'max_iter': [500]
}

# Decision tree clasifier (DTC)
dtc_grid = {
    'criterion': ['gini', 'entropy'],
    'max_features': [None],
    'class_weight': [None],
    'min_samples_leaf': [5, 50, 100],
    'min_samples_split': [10, 100, 200]
}

# Random forest classifier (RFC)
rfc_grid = {
    'n_estimators': [3, 5, 10, 100],
    'criterion': ['gini'],
    'max_features': [None],
    'class_weight': [None],
    'min_samples_leaf': [5, 50, 100],
    'min_samples_split': [5, 50]
}

# Support vector machine (SVM)
svm_grid = {
    'C': [1.0],
    'kernel': ['rbf'],
    'gamma': [1.0],
    'probability': [True],
    'max_iter': [-1],
}


# Artificial neural network (ANN)
ann_grid = {
    'hidden_layer_sizes': [(10,), (50,)],
    'activation': ['logistic'],
    'solver': ['adam'],
    'alpha': [0.0001],
    'batch_size': ['auto'],
    'learning_rate': ['constant'],
    'learning_rate_init': [0.001],
    'power_t': [0.5],
    'max_iter': [1000],
    'tol': [1e-4],
    'warm_start': [False],
    'momentum': [0.9],
}

# Create default grid
_DEFAULT_ESTIMATOR_GRID = {
    'gnb': gnb_grid,
    'dtc': dtc_grid,
    'rfc': rfc_grid,
    'svm': svm_grid,
    'ann': ann_grid
}

_DEFAULT_FEATURES = {
    '6': sorted(['alp', 'alt', 'bil', 'cre', 'crp', 'wbc']),
    '7': sorted(['alp', 'alt', 'bil', 'cre', 'crp', 'wbc', 'pct']),
    '21': sorted(['alb', 'alp', 'alt', 'baso', 'bil', 'cl', 'cre', 'crp', 'egfr',
                  'eos', 'k', 'ly', 'mcv', 'mono', 'mpv', 'nrbca', 'plt', 'rbc',
                  'rdw', 'urea', 'wbc']),
    '27': sorted(['alb', 'alp', 'alt', 'baso', 'bil', 'cl', 'cre', 'crp', 'egfr',
                  'eos', 'hct', 'hgb', 'k', 'ly', 'mch', 'mchc', 'mcv', 'mono',
                  'mpv', 'neut', 'nrbca', 'plt', 'rbc', 'rdw', 'sodium', 'urea', 'wbc'])
}

# ------------------------------
# data constants
# ------------------------------
# Define feature group
slug = '21'

# Feature vector
features = _DEFAULT_FEATURES[slug]

# Targets
target = 'micro_confirmed'
compare = 'covid_confirmed'

# ------------------------------
# pipeline constants
# ------------------------------
# Folder name
outpath = './outputs'

# ------------------------------
# load data
# ------------------------------
# Load data
data = pd.read_csv('./data/covid19/daily_profiles.csv')
data.columns = [c.lower() for c in data.columns]
data = data[features + [target, compare]]
data.covid_confirmed = data.covid_confirmed.astype(int)
data.micro_confirmed = data.micro_confirmed.astype(int)

print(data.columns)

# Get x and y (y is fake)
X_aux = data[features + [target, compare]]
y_aux = data[target]

from pyPI.preprocessing.splitters import PipelineSplitter

m = PipelineSplitter().split(X_aux.to_numpy(), y_aux.to_numpy())


#Una extra
CVS = pd.DataFrame(m['CVS'][:,:-1], columns=features + [target, compare])
HOS = pd.DataFrame(m['HOS'][:,:-1], columns=features + [target, compare])


# ------------------------------
# create pipeline
# ------------------------------
# Create the objects
flt = 'none'
imp = 'median'
smp = 'smote'
pre = 'std'
spl = 'kfold'

# ------------------------------
# create targets to compute
# ------------------------------
# Create targets
estimators = ['ann']

# Create tests
tests = ['HOS']

# Train
train = True

# ------------------------------
# loop computing pipelines
# ------------------------------
# For each site
for estimator in estimators:

    # Create pipeline path
    pipelinepath = '%s/%s-%s-%s-%s-%s-%s-%s' % \
                   (outpath, estimator, flt, imp, smp, pre, spl, slug)

    # Create pipeline
    pipeline = Pipeline(steps=[('filter', flt),
                               ('imputer', imp),
                               ('splitter', spl),
                               ('sampler', smp),
                               ('scaler', pre),
                               ('estimator', estimator)],
                        outpath=pipelinepath,
                        hos_data=HOS[features + [target]].to_numpy(),
                        verbose=5)

    # --------------
    # Train
    # --------------
    if train:
        # Create the parameters for a single estimator.
        estimator_kwgs = {}

        # Create the parameters to perform a grid search for several estimators.
        # Since all of them are going to be trained and tested with the same
        # portions of data, it will me more accurate the comparison between
        # their performances as when using different portions of data which are
        # selected randomly.
        estimator_grid = _DEFAULT_ESTIMATOR_GRID[estimator]

        # Create a folder in which all the dataset partitions used within the
        # pipeline are saved as numpy arrays. The partitions created calling
        # this method are randomised. The datasets are HOS, HOSbal, CVS and Fn.
        # This method returns X_cvs and y_cvs.
        #pipeline.prep(CVS[features], CVS[target])

        # Fit the grid search estimator model
        pipeline.fit(X=CVS[features].to_numpy(),
                     y=CVS[target].to_numpy(),
                     estimator_grid=estimator_grid)

    # --------------
    # Test
    # --------------
    # Evaluate on the following sets
    #pipeline.evaluate_data(X=HOS[features].to_numpy(),
    #                       y=HOS[target].to_numpy())

    estm = pickle.load(open('%s/estimators/iteration_00/estimator_00.pkl' % pipelinepath, 'rb'))

    pred = estm.predict_proba(HOS[features].to_numpy())

    print(estm)
    print(pred)

    results = pd.DataFrame()
    results['p1'] = pred[:,1]
    results['covid_confirmed'] = HOS['covid_confirmed']
    results['micro_confirmed'] = HOS['micro_confirmed']
    results['target'] = target
    results['type'] = None


    results.loc[(results['p1']>0.5) & results[target]==1, 'type'] = 'TP'
    results.loc[(results['p1']>0.5) & results[target]==0, 'type'] = 'FP'
    results.loc[(results['p1']<0.5) & results[target]==1, 'type'] = 'FN'
    results.loc[(results['p1']<0.5) & results[target]==0, 'type'] = 'TN'

    results.to_csv('%s/results-%s.csv' % (pipelinepath, target))

    import sys
    sys.exit()

    # --------------
    # Compare
    # --------------
    X_hos = pd.DataFrame(hos, columns=data.columns)[features]
    y_hos = pd.DataFrame(hos, columns=data.columns)[compare]
    
    # Load estimator
    estm = pickle.load(open('%s/estimators/iteration_00/estimator_00.pkl', 'rb'))

    print(estm)








import sys
sys.exit()

# Create matrices
X = data[features].to_numpy()
y = data[target].to_numpy()



import sys

sys.exit()

# Display estimator
print("\nEstimator: \n %s" % estm_micro)

data_micro = np.load('/tmp/123.npy')

# Predict probabilities
# probs = estm.predict_proba(data)

#
# print(probs)
