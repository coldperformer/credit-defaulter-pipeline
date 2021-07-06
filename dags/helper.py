import os
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from imblearn import over_sampling


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, colnames, targetName, n_fold=5, verbosity=True, discardOriginal_col=False):
        """Initialize all the important variables here.

        Arguments:
        ----------

        colnames (list): The column names of the dataset.
        targetName (str): The target column of the dataset.
        n_fold (int): The number of folds to perform target encoding.
        verbosity (int): To view logs of the execution.
        discardOriginal_col (bool): True if remove the original feature values else False.
        """
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col

    def fit(self, X, y=None):
        """Fits the data using input features and target feature."""
        return self

    def transform(self, X):
        """Transforms the data using input features."""
        # Unit testing of the specified arguments.
        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=42)
        col_mean_name = self.colnames + '_' + 'Enc'
        X[col_mean_name] = np.nan
        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(
                X_tr.groupby(self.colnames)[self.targetName].mean())
            X[col_mean_name].fillna(mean_of_target, inplace=True)
        if self.verbosity:
            encoded_feature = X[col_mean_name].values
        if self.discardOriginal_col:
            X = X.drop(self.colnames, axis=1)
        return X


class CreditDefaulterHelper:

    def __init__(self, check_load_path, check_save_path, num_save_path, cat_save_path, feature_saveXy, savemodelpath):
        """Initialize all the important variables here."""

        self.cat_cols = ['year', 'state', 'own_type', 'income_type', 'app_type',
                         'loan_purpose', 'interest_payments', 'grade', 'loan_duration', 'is_default']
        self.num_cols = ['emp_duration', 'annual_pay', 'loan_amount', 'interest_rate',
                         'dti', 'total_pymnt', 'total_rec_prncp', 'recoveries', 'installment']

        self.check_load_path = check_load_path
        self.check_save_path = check_save_path
        self.num_save_path = num_save_path
        self.cat_save_path = cat_save_path
        self.feature_saveXy = feature_saveXy
        self.savemodelpath = savemodelpath

    def checkFileDropper(self, features=None):
        """Validate the file location, drop unncessary faetures, and then save the data.

        Arguments:
        ----------

        features (list): A list of features to drop from the dataset.
        """
        data = pd.read_csv(filepath_or_buffer=self.check_load_path)
        if len(data) != 0:
            
            print('File Found! Performing operations...')
            data.drop(labels=features, axis=1, inplace=True)
            data.to_csv(path_or_buf=self.check_save_path, index=False)

        else:
            raise Exception("File Not found! Check logs...")
        return 'Dropping Unncessary Features Success!'

    def numericTransformer(self):
        """Performs numerical transformations over the raw input data."""
    
        if os.path.exists(path=self.check_save_path):

            print('Raw file found! Performing numerical transformations...')
            data = pd.read_csv(filepath_or_buffer=self.check_save_path, usecols=self.num_cols)

            # Replacing zeros with the middle value of the dataset
            data['dti'] = data['dti'].replace(to_replace=0, value=np.median(data['dti']))

            # Outiler Handling: Removing outliers in the dataset
            out_features = ['annual_pay', 'loan_amount', 'interest_rate', 'dti',
                            'total_pymnt', 'total_rec_prncp', 'recoveries', 'installment']

            for col in out_features:
                outlier_index = data[data[col] > data[col].quantile(0.99)].index
                data.loc[outlier_index, col] = data[col].quantile(0.99)

            data.to_csv(path_or_buf=self.num_save_path, index=False)
        else:
            raise Exception("File Not found! Check logs...")
        return 'Numerical Transformations Success!'

    def categoryTransformer(self, feed_dict=dict()):
        """Performs categorical transformations over the raw input data.

        Arguments:
        ----------

        feed_dict (dict): The dictionary containing type of encoder.
                          The key values should be either LabelEncoder, or
                          OneHotEncoder, or KFoldTargetEncoding, or all.
        """
        if os.path.exists(path=self.check_save_path):

            data = pd.read_csv(filepath_or_buffer=self.check_save_path, usecols=self.cat_cols)

            print('Raw file found! Performing categorical transformations...')

            ## Feature Encoding: Encoding categorical features
            for key, value in feed_dict.items():
                if key == "LabelEncoder":
                    encoder = LabelEncoder()
                    for col in value:
                        data[col] = encoder.fit_transform(data[col])

                elif key == "OneHotEncoder":
                    data = pd.get_dummies(data=data, columns=value)

                elif key == "KFoldTargetEncoding":
                    for col in list(value):
                        kfold_te = KFoldTargetEncoder(
                            colnames=col, targetName='is_default', discardOriginal_col=True)
                        data = kfold_te.fit_transform(X=data)

            data.to_csv(path_or_buf=self.cat_save_path, index=False)
        else:
            raise Exception("File Not found! Check logs...")
        return 'Categorical Transformation Success!'

    def featuresMergeScaler(self):
        """Perform feature scaling, SMOTE over minority class and save final dataset."""
        if (os.path.exists(path=self.num_save_path) & os.path.exists(path=self.cat_save_path)):

            ## Loading numerical and categorical featured data
            datacat = pd.read_csv(filepath_or_buffer=self.cat_save_path)
            datanum = pd.read_csv(filepath_or_buffer=self.num_save_path)

            print('Data files found! Performing operations...')
    
            ## Features Merging: Concatenating numerical and categorical features at column level
            datacombined = pd.concat(objs=[datacat, datanum], axis=1, verify_integrity=True)

            # Dropping unnecessary zeros because annual_pay can't be zero
            datacombined = datacombined.drop(index=datacombined[datacombined['annual_pay'] == 0].index, axis=0)

            ## Data Splitting: Splitting data into input and target features.
            X = datacombined.drop(labels=['is_default'], axis=1)
            y = datacombined['is_default']

            ## Feature Scaling: Performing standardization over input features
            scaler = StandardScaler()
            scale_fit = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(data=scale_fit, columns=X.columns)

            ## SMOTE: Performing minority oversampling
            sm = over_sampling.SMOTE(random_state=42, sampling_strategy='minority', n_jobs=-1)
            Xnew, y = sm.fit_resample(X_scaled, y)

            ## Saving final input and target
            data_new = pd.DataFrame(data=Xnew, columns=X.columns)
            data_new = pd.concat(objs=[data_new, y], axis=1, verify_integrity=True)
            data_new.to_csv(path_or_buf=self.feature_saveXy, index=False)
        else:
            raise Exception("File Not found! Check logs...")

        return 'Features Merging and Scaling Success!'

    def modelDeveloper(self, algorithm=None):
        """Perform model development over specified algorithm and save into pickle file.

        Arguments:
        ----------

        algorithm (str): Supporting models to build: ("Logistic", "RandomForest", "XGBoost")                  
        """

        if (os.path.exists(path=self.feature_saveXy)):

            data = pd.read_csv(filepath_or_buffer=self.feature_saveXy)

            print('Data file found! Performing operations...')

            ## Loading input and target featured data
            X = data.drop(labels=['is_default'], axis=1)
            y = data['is_default']

            ## Model Development: Develops model based on the specified parameter
            if algorithm == 'Logistic':
                clf = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
                clf.fit(X, y)
                with open(self.savemodelpath+'/log.pckl', 'wb') as file:
                    pickle.dump(obj=clf, file=file)

            elif algorithm == "RandomForest":
                clf = RandomForestClassifier(max_depth=5, random_state=42, n_jobs=-1, class_weight='balanced')
                clf.fit(X, y)
                with open(self.savemodelpath+'/random_forest.pckl', 'wb') as file:
                    pickle.dump(obj=clf, file=file)

            elif algorithm == "XGBoost":
                clf = XGBClassifier(max_depth=7, n_jobs=-1, random_state=42, use_label_encoder=False, eval_metric='logloss')
                clf.fit(X, y)
                with open(self.savemodelpath+'/xgb.pckl', 'wb') as file:
                    pickle.dump(obj=clf, file=file)
        else:
            raise Exception("File Not found! Check logs...")

        return 'Model Development Success!'
