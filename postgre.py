import pandas as pd
import numpy as np
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

connection = psycopg2.connect(
    host="branchhomeworkdb.cv8nj4hg6yra.ap-south-1.rds.amazonaws.com",
    port=5432,
    user="datascientist",
    password="47eyYBLT0laW5j9U24Uuy8gLcrN",
    database="branchdsprojectgps"
)

loan_outcomes_query = "SELECT * FROM loan_outcomes;"
gps_fixes_query = "SELECT * FROM gps_fixes;"
user_attributes_query = "SELECT * FROM user_attributes;"

loan_outcomes = pd.read_sql_query(loan_outcomes_query, connection)
gps_fixes = pd.read_sql_query(gps_fixes_query, connection)
user_attributes = pd.read_sql_query(user_attributes_query, connection)

connection.close()

loan_outcomes.columns = [col.lower() for col in loan_outcomes.columns]
gps_fixes.columns = [col.lower() for col in gps_fixes.columns]
user_attributes.columns = [col.lower() for col in user_attributes.columns]

loan_outcomes['application_at'] = pd.to_datetime(loan_outcomes['application_at'])
gps_fixes['gps_fix_at'] = pd.to_datetime(gps_fixes['gps_fix_at'])

gps_features = gps_fixes.groupby('user_id').agg(
    {
        'latitude': ['mean', 'std', 'min', 'max'],
        'longitude': ['mean', 'std', 'min', 'max'],
        'gps_fix_at': ['count']
    }
)
gps_features.columns = ['_'.join(col) for col in gps_features.columns]


