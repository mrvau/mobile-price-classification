import pandas as pd
import numpy as np
import pickle
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score,classification_report

path = kagglehub.dataset_download("iabhishekofficial/mobile-price-classification")

print("Path to dataset files:", path)

df_train = pd.read_csv(f"{path}/train.csv")
df_test = pd.read_csv(f"{path}/test.csv")
df_test.drop(columns=["id"], inplace=True)

print(df_train.shape)
print(df_test.shape)

X = df_train.drop(columns=["price_range"])
y = df_train["price_range"]

print(X.shape)
print(y.shape)

num_cols = X.select_dtypes(["int64", "float64"]).columns
cat_cols = X.select_dtypes(["object"]).columns

scaler = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

encoder = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", scaler, num_cols),
        ("cat", encoder, cat_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rfc = RandomForestClassifier(n_estimators=305, min_samples_split=4, max_depth=47, random_state=42)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rfc)
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
recall = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')
f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')

report = classification_report(y_true=y_test, y_pred=y_pred)

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Classification Report:\n{report}")

with open("mobile_price.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("✅ Random Forest pipeline saved as mobile_price.pkl")