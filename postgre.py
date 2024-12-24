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


user_data = loan_outcomes.merge(user_attributes, on='user_id', how='left')
user_data = user_data.merge(gps_features, on='user_id', how='left')

categorical_cols = user_data.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in categorical_cols:
    user_data[col] = encoder.fit_transform(user_data[col])

user_data['days_since_loan'] = (user_data['application_at'].max() - user_data['application_at']).dt.days

user_data['income_to_age_ratio'] = user_data['cash_incoming_30days'] / (user_data['age'] + 1)
user_data['gps_fix_density'] = user_data['gps_fix_at_count'] / user_data['days_since_loan']

user_data.replace([np.inf, -np.inf], np.nan, inplace=True)
user_data.fillna(user_data.select_dtypes(include=['number']).median(), inplace=True)

X = user_data.drop(columns=['user_id', 'application_at', 'loan_outcome'])
y = user_data['loan_outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
log_reg = LogisticRegression(random_state=42)

ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model),
        ('xgb', xgb_model),
        ('logreg', log_reg)
    ],
    voting='soft'
)

ensemble_model.fit(X_train, y_train)

y_pred = ensemble_model.predict(X_test)
y_prob = ensemble_model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC: {roc_auc}")

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

feature_importances = pd.DataFrame(
    {'feature': X.columns, 'importance': np.mean([model.feature_importances_ for name, model in ensemble_model.named_estimators_.items() if hasattr(model, 'feature_importances_')], axis=0)}
).sort_values(by='importance', ascending=False)

plt.barh(feature_importances['feature'], feature_importances['importance'])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances")
plt.show()