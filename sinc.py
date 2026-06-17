import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv("synthetic_soc_1000_indexed.csv")
sampledata = df.sample(n=600, random_state=42).copy()

print("Original shape:", sampledata.shape)
print(sampledata.head(3))


numeric = sampledata.select_dtypes(include=[np.number]).columns
categorical = sampledata.select_dtypes(include=['object']).columns

median_vals = sampledata[numeric].median()
sampledata[numeric] = sampledata[numeric].fillna(median_vals)

for col in categorical:
    modes = sampledata[col].mode()
    fill_val = modes.iloc[0] if not modes.empty else 'Unknown'  
    sampledata[col] = sampledata[col].fillna(fill_val)


missing_pct = (sampledata.isnull().sum() / len(sampledata) * 100).round(2)
print("Missing %:\n", missing_pct[missing_pct > 0])
print("Total missing after:", sampledata.isnull().sum().sum())  


label_encoders = {}
for col in categorical:
    le = LabelEncoder()
    sampledata[col] = le.fit_transform(sampledata[col].astype(str))
    label_encoders[col] = le


label = LabelEncoder()
sampledata["label_encoded"] = label.fit_transform(sampledata["label"])
X = sampledata.drop(['index', 'label', 'label_encoded'], axis=1)
y = sampledata["label_encoded"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
sfs = SequentialFeatureSelector(
    model, 
    n_features_to_select=10, 
    direction='forward', 
    cv=5, 
    scoring='accuracy'
)
sfs.fit(X_train, y_train)

print("Selected features:", X.columns[sfs.get_support()].tolist())
print("CV score:", sfs.cv_results_['mean_test_score'][-1].round(3))

X_train_selected = sfs.transform(X_train)
print("Selected shape:", X_train_selected.shape)






