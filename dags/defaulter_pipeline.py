"""
# Credit Card Defaulter dag

- This dag is a prototype of ETL execution of credit defaulters data sourced from Kaggle.

- The data has been migrated to a CSV table using which the entire pipeline was prepared and formulated.

![img](https://i0.wp.com/blog.bankbazaar.com/wp-content/uploads/2016/03/Surviving-a-Credit-Card-Default-thumb-nail.png?fit=435%2C267&ssl=1)
"""
from airflow import DAG
from pendulum import timezone
from datetime import datetime
from datetime import timedelta
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator
from helper import CreditDefaulterHelper


local_tz = timezone(name='Asia/Kolkata')

default_args = {
    'owner': 'coldperformer',
    'start_date': datetime(year=2021, month=6, day=28, hour=15, minute=15, tzinfo=local_tz), 
    'end_date': datetime(year=2021, month=7, day=25, hour=15, minute=15, tzinfo=local_tz),
}

# Define all the connections here
check_load_path = 'https://storage.googleapis.com/industryanalytics/LoanDefaultData.csv'
check_save_path = 'saves/01 newraw'+'/newraw.csv'
num_save_path = 'saves/02 catnumdata/numeric'+'/numericdata.csv'
cat_save_path = 'saves/02 catnumdata/categoric'+'/categoricaldata.csv'
feature_saveXy = 'saves/03 finaldata'+'/finaldata.csv'
savemodelpath = 'saves/04 models'

# Initiating a new object/variable for ETLCarsPipeline
obj = CreditDefaulterHelper(check_load_path, check_save_path, num_save_path, cat_save_path, feature_saveXy, savemodelpath)

with DAG(dag_id='Credit_Defaulter', default_args=default_args, schedule_interval="*/25 * * * *", catchup=False) as dag:
    dag.doc_md = __doc__

    # Task 1-------------------------------------------------------------------
    connect = PythonOperator(python_callable=obj.checkFileDropper , task_id='Connection_Check', 
    op_kwargs={'features': ['cust_id', 'date_issued', 'date_final']})

    connect.doc_md = """"""

    # Task 2-------------------------------------------------------------------
    trans_numeric = PythonOperator(python_callable=obj.numericTransformer, task_id='Transform_Numeric_Data')

    trans_numeric.doc_md = """"""

    # Task 3-------------------------------------------------------------------
    trans_category = PythonOperator(python_callable=obj.categoryTransformer, task_id='Transform_Category_Data', 
    op_kwargs={
        'feed_dict': {
            'LabelEncoder': ['year'], 
            'OneHotEncoder': ['own_type', 'income_type', 'app_type', 'interest_payments', 'grade', 'loan_duration'],
            'KFoldTargetEncoding': ['state', 'loan_purpose']
        }
    })

    trans_category.doc_md = """"""

    # Task 4-------------------------------------------------------------------
    merge_scaler = PythonOperator(python_callable=obj.featuresMergeScaler, task_id='Merge_Drop_Data', provide_context=True)

    merge_scaler.doc_md = """"""

    # Task 5-------------------------------------------------------------------
    logisticmodel = PythonOperator(python_callable=obj.modelDeveloper, task_id='logistic_regression', 
    op_kwargs={'algorithm': 'Logistic'})

    logisticmodel.doc_md = """"""

    # Task 6-------------------------------------------------------------------
    randomforestmodel = PythonOperator(python_callable=obj.modelDeveloper, task_id='random_forest_model', 
    op_kwargs={'algorithm': 'RandomForest'})

    randomforestmodel.doc_md = """"""

    # Task 7-------------------------------------------------------------------
    xgbmodel = PythonOperator(python_callable=obj.modelDeveloper, task_id='xgb_model', 
    op_kwargs={'algorithm': 'XGBoost'})

    xgbmodel.doc_md = """"""


# Setting tasks dependencies
connect >> [trans_numeric, trans_category] >> merge_scaler >> [logisticmodel, randomforestmodel, xgbmodel]
