#!/usr/bin/python3

from helper import CreditDefaulterHelper


check_load_path = 'https://storage.googleapis.com/industryanalytics/LoanDefaultData.csv'
check_save_path = 'saves/01 newraw'+'/newraw.csv'
num_save_path = 'saves/02 catnumdata/numeric'+'/numericdata.csv'
cat_save_path = 'saves/02 catnumdata/categoric'+'/categoricaldata.csv'
feature_saveXy = 'saves/03 finaldata'+'/finaldata.csv'
savemodelpath = 'saves/04 models'

obj = CreditDefaulterHelper(check_load_path, 
                            check_save_path,
                            num_save_path, 
                            cat_save_path, 
                            feature_saveXy, 
                            savemodelpath)

# Unit testing
assert obj.checkFileDropper(features=['cust_id', 'date_issued', 'date_final']) == 'Dropping Unncessary Features Success!'
assert obj.numericTransformer() == 'Numerical Transformations Success!'
assert obj.categoryTransformer(
    feed_dict={
            'LabelEncoder': ['year'], 
            'OneHotEncoder': ['own_type', 'income_type', 'app_type', 'interest_payments', 'grade', 'loan_duration'],
            'KFoldTargetEncoding': ['state', 'loan_purpose']
        }) == 'Categorical Transformation Success!'

assert obj.featuresMergeScaler() == 'Features Merging and Scaling Success!'
assert obj.modelDeveloper(algorithm='Logistic') == 'Model Development Success!'
assert obj.modelDeveloper(algorithm='RandomForest') == 'Model Development Success!'
assert obj.modelDeveloper(algorithm='XGBoost') == 'Model Development Success!'

print('Testing Success!')
