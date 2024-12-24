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


