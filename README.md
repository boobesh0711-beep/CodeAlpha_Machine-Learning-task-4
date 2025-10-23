# CodeAlpha_Machine-Learning-task-4
Disease recognition 
# -------------------------------------------
# Disease Prediction from Medical Data
# CodeAlpha Internship – Task 4
# -------------------------------------------

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------
# Step 2: Load Dataset
# (Heart Disease dataset from UCI Repository)
# -------------------------------------------
url = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
df = pd.read_csv(url)

print("✅ Dataset Loaded Successfully!")
display(df.head())

# -------------------------------------------
# Step 3: Data Exploration
# -------------------------------------------
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistical Summary:")
display(df.describe())

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------------------------------
# Step 4: Data Preprocessing
# -------------------------------------------
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------
# Step 5: Model Training
# -------------------------------------------

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# -------------------------------------------
# Step 6: Model Evaluation
# -------------------------------------------

# Logistic Regression
y_pred_lr = lr.predict(X_test_scaled)
print("\n🔹 Logistic Regression Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr)*100, 2), "%")
print(classification_report(y_test, y_pred_lr))

# Random Forest
y_pred_rf = rf.predict(X_test)
print("\n🔹 Random Forest Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf)*100, 2), "%")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------------------
# Step 7: Prediction on New Data
# -------------------------------------------

sample = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])  # Example
prediction = rf.predict(sample)
print("\nSample Prediction Output:")
print("Disease Detected" if prediction[0]==1 else "No Disease Detected")

# -------------------------------------------
# Step 8: Conclusion
# -------------------------------------------
print("\n✅ Project Completed Successfully!")
print("Best Performing Model: Random Forest Classifier")
