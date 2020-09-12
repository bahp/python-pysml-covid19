# Generic
import pandas as pd

# Specific
from pathlib import Path

# ----------------------
# Load data
# ----------------------
# Create path
path = Path('./data/clwsql008-susceptibility/')

# Create dataframe
micro = pd.DataFrame()

# List files
for f in list(path.glob('**/*.csv')):
    micro = micro.append(pd.read_csv(f))

# Drop duplicates
micro.drop_duplicates(inplace=True)

# Pivote
micro = micro[['PtNumber', 'ReceiveDate', 'BatTstCode', 'FinalDate']]
micro = micro.drop_duplicates()
micro.reset_index(inplace=True, drop=True)
micro['micro_confirmed'] = True

# ------------------
# Merge with PIMS
# ------------------
# Read
folder = './data/ipc-sarscov-inpatients-sitrep-20200806'
filepath_pims = '%s/pims-patients.csv' % folder
pims = pd.read_csv(filepath_pims)
pims = pims[['caseHospitalId', 'caseNHSNumber']]

# Merge with pims
micro = micro.merge(pims, how='inner',
                          left_on='PtNumber',
                          right_on='caseHospitalId')
micro = micro.drop_duplicates()

print(micro)

# Save
micro.to_csv('./outputs/_datasets/daily_profiles_microbiology.csv', index=False)

