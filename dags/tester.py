#!/usr/bin/python3

import os
from helper import CreditDefaulterHelper

checksavedir = 'saves/01 newraw' 
numsavedir = 'saves/02 catnumdata/numeric'
catsavedir = 'saves/02 catnumdata/categoric'
featurexydir = 'saves/03 finaldata'
savemodeldir = 'saves/04 models'

# Create new directories if not exists
paths = [checksavedir, numsavedir, catsavedir, featurexydir, savemodeldir]

for path in paths:
    if not os.path.exists(path=path):
        os.makedirs(path)
        print('New Path Created: ', path)
    else:
        print('New Path Creation Not Required.')

check_load_path = 'https://storage.googleapis.com/industryanalytics/LoanDefaultData.csv'
check_save_path = checksavedir + '/newraw.csv'
num_save_path = numsavedir + '/numericdata.csv'
cat_save_path = catsavedir + '/categoricaldata.csv'
feature_saveXy = featurexydir + '/finaldata.csv'
savemodelpath = savemodeldir

obj = CreditDefaulterHelper(check_load_path, 
                            check_save_path,
                            num_save_path, 
                            cat_save_path, 
                            feature_saveXy, 
                            savemodelpath)

# Unit testing
assert obj.checkFileDropper(features=['cust_id', 'date_issued', 'date_final']) == 'Dropping Unncessary Features Success!'
print('Task 1 Success!, Unit Test 1 Passed')
assert obj.numericTransformer() == 'Numerical Transformations Success!'
print('Task 2 Success!, Unit Test 2 Passed')
assert obj.categoryTransformer(
    feed_dict={
            'LabelEncoder': ['year'], 
            'OneHotEncoder': ['own_type', 'income_type', 'app_type', 'interest_payments', 'grade', 'loan_duration'],
            'KFoldTargetEncoding': ['state', 'loan_purpose']
        }) == 'Categorical Transformation Success!'
print('Task 3 Success!, Unit Test 3 Passed')

assert obj.featuresMergeScaler() == 'Features Merging and Scaling Success!'
print('Task 4 Success!, Unit Test 4 Passed')
assert obj.modelDeveloper(algorithm='Logistic') == 'Model Development Success!'
print('Task 5 Success!, Unit Test 5 Passed')
assert obj.modelDeveloper(algorithm='RandomForest') == 'Model Development Success!'
print('Task 6 Success!, Unit Test 6 Passed')
assert obj.modelDeveloper(algorithm='XGBoost') == 'Model Development Success!'
print('Task 7 Success!, Unit Test 7 Passed')

print('All Tests Passed!')
