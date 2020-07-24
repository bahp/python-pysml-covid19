# Libraries
import pandas as pd
import numpy as np

# Read
df1 = pd.read_csv('data/covid19/daily_profiles.csv')
df2 = pd.read_csv('./data/covid19/one_line_values_pathology.csv')

# Rename
df1 = df1.rename(columns={'covid_confirmed': 'covid_confirmed1'})
df2 = df2.rename(columns={'covid_confirmed': 'covid_confirmed2',
                          'dateResult': 'date_result'})

print(np.sum(df1.covid_confirmed1))
print(np.sum(df2.covid_confirmed2))

# Show
print(df1.columns)
print(df2.columns)

df1 = df1[['_uid', 'date_result', 'covid_confirmed1']]
df2 = df2[['_uid', 'date_result', 'covid_confirmed2']]

df1.set_index(['_uid', 'date_result'], inplace=True)
df2.set_index(['_uid', 'date_result'], inplace=True)



df = df1.merge(df2, how='outer', on=['_uid', 'date_result'])

print(df)
import sys
sys.exit()
# Outcomes
df1 = df1[[]]

print(df1.shape)
print(df2.shape)