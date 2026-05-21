import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Load dataset
df = pd.read_csv("employee_salary_dataset.csv")

# Drop unnecessary columns
df.drop(["EmployeeID", "Name"], axis=1, inplace=True)

# Encode categorical columns
label_encoders = {}

categorical_columns = [
    "Department",
    "Education_Level",
    "Gender",
    "City"
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop("Monthly_Salary", axis=1)
y = df["Monthly_Salary"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
score = r2_score(y_test, predictions)

print("Model Accuracy:", score)

# Save model
joblib.dump(model, "salary_model.pkl")

# Save encoders
joblib.dump(label_encoders, "encoders.pkl")

print("Model Saved Successfully")