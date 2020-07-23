###############################################################################
# Author: Bernard Hernandez
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
import warnings
import numpy as np
import pandas as pd

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
    'C': [0.1, 1.0, 0.01],
    'kernel': ['rbf'],
    'gamma': [0.1, 1.0, 0.01],
    'probability': [True],
    'max_iter': [-1],
}

# Artificial neural network (ANN)
ann_grid = {
    'hidden_layer_sizes': [(1,), (10,), (50,),
                           (5, 5), (10, 10), (5, 5, 5)],
    'activation': ['logistic', 'relu'],
    'solver': ['adam'],
    'alpha': [1., 0.1, 0.0001],
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
    '27': sorted(['alb', 'alp', 'alt', 'baso', 'bil', 'cl', 'cre', 'crp', 'egfr',
                  'eos', 'hct', 'hgb', 'k', 'ly', 'mch', 'mchc', 'mcv', 'mono',
                  'mpv', 'neut', 'nrbca', 'plt', 'rbc', 'rdw', 'sodium', 'urea', 'wbc'])
}
# ------------------------------
# data constants
# ------------------------------
# Define feature group
slug = '6'

# Feature vector
features = _DEFAULT_FEATURES[slug]
target = 'covid_confirmed'

# ------------------------------
# pipeline constants
# ------------------------------
# Folder name
outpath = './outputs'

# ------------------------------
# load data
# ------------------------------
# Load data
data = pd.read_csv('./data/covid19/one_line_values_pathology.csv')
data.columns = [c.lower() for c in data.columns]

# Create matrices
X = data[features].to_numpy()
y = data[target].to_numpy()

# ------------------------------
# create pipeline
# ------------------------------
# Create the objects
flt = 'none'
imp = 'median'
smp = 'smote'
pre = 'std'
spl = 'skfold'

# ------------------------------
# create targets to compute
# ------------------------------
# Create targets
estimators = ['svm']

# Create tests
tests = ['HOS', 'HOSbal']

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
                        hos_size=0.25,
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
        pipeline.prep(X, y)

        # Fit the grid search estimator model
        pipeline.fit(X=None, y=None, name='CVS', estimator_grid=estimator_grid)

    # --------------
    # Test
    # --------------
    # Evaluate on the following sets
    pipeline.evaluate_data(names=tests)

# -----------------------
# The final results
# -----------------------
# Note that that the pipeline creates a folder with several objects.
# These objects are briefly explained below. For more information
# see url.

# Structure:
# - folder
#   |- data
#      |- CSV
#      |- HOS
#      |- HOSbal
#      |- Fn
#   |- estimators
#   |- iterations
#      |- iteration_00.pkl
#      |- iteration_0n.pkl
#   |- summaries
#      |- complete
#         |- summary.csv
#         |- summmean.csv
#         |- summstd.csv
#      |- partial
#         |- summary_01.csv
#         |- summary_0n.csv
