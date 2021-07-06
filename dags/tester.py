#!/usr/bin/python3

# Importing Libraries
import os
import time
from helper import CreditDefaulterHelper

# Setting up path variables
check_load_path = 'https://storage.googleapis.com/industryanalytics/LoanDefaultData.csv'
checksavedir = 'saves/01 newraw' 
numsavedir = 'saves/02 catnumdata/numeric'
catsavedir = 'saves/02 catnumdata/categoric'
featurexydir = 'saves/03 finaldata'
savemodeldir = 'saves/04 models'

# Create new directories at specified paths if not exists
paths = [checksavedir, numsavedir, catsavedir, featurexydir, savemodeldir]

for path in paths:
    if not os.path.exists(path=path):
        os.makedirs(path)
        print(path, 'New path created!')
    else:
        print('New path creation not required.')

obj = CreditDefaulterHelper(check_load_path=check_load_path, 
                            check_save_path=checksavedir + '/newraw.csv',
                            num_save_path=numsavedir + '/numericdata.csv', 
                            cat_save_path=catsavedir + '/categoricaldata.csv', 
                            feature_saveXy=featurexydir + '/finaldata.csv', 
                            savemodelpath=savemodeldir)

# Unit testing: Test 1
start = time.time()
assert obj.checkFileDropper(features=['cust_id', 'date_issued', 'date_final']) == 'Dropping Unncessary Features Success!'
end = time.time()
print('Test 1 Success!, Execution time: ', end-start)

# Unit testing: Test 2
start = time.time()
assert obj.numericTransformer() == 'Numerical Transformations Success!'
end = time.time()
print('Test 2 Success!, Execution time: ', end-start)

# Unit testing: Test 3
start = time.time()
assert obj.categoryTransformer(
    feed_dict={
        'LabelEncoder': ['year'], 
        'OneHotEncoder': ['own_type', 'income_type', 'app_type', 
        'interest_payments', 'grade', 'loan_duration'],
        'KFoldTargetEncoding': ['state', 'loan_purpose']
        }) == 'Categorical Transformation Success!'
end = time.time()
print('Test 3 Success!, Execution time: ', end-start)

# Unit testing: Test 4
start = time.time()
assert obj.featuresMergeScaler() == 'Features Merging and Scaling Success!'
end = time.time()
print('Test 4 Success!, Execution time: ', end-start)

# Unit testing: Test 5
start = time.time()
assert obj.modelDeveloper(algorithm='Logistic') == 'Model Development Success!'
end = time.time()
print('Test 5 Success!, Execution time: ', end-start)

# Unit testing: Test 6
start = time.time()
assert obj.modelDeveloper(algorithm='RandomForest') == 'Model Development Success!'
end = time.time()
print('Test 6 Success!, Execution time: ', end-start)

# Unit testing: Test 7
start = time.time()
assert obj.modelDeveloper(algorithm='XGBoost') == 'Model Development Success!'
end = time.time()
print('Test 6 Success!, Execution time: ', end-start)
