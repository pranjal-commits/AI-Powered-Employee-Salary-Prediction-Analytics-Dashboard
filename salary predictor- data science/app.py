import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("employee_salary_dataset.csv")

model = joblib.load("salary_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("AI-Powered Employee Salary Predictor")

st.write("Predict employee salaries using Machine Learning")

# SIDEBAR

st.sidebar.header("Employee Details")

# User Inputs
department = st.sidebar.selectbox(
    "Department",
    encoders["Department"].classes_
)

experience = st.sidebar.number_input(
    "Years of Experience",
    min_value=0,
    max_value=40
)

education = st.sidebar.selectbox(
    "Education Level",
    encoders["Education_Level"].classes_
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=65
)

gender = st.sidebar.selectbox(
    "Gender",
    encoders["Gender"].classes_
)

city = st.sidebar.selectbox(
    "City",
    encoders["City"].classes_
)

# Encode inputs
department_encoded = encoders["Department"].transform([department])[0]
education_encoded = encoders["Education_Level"].transform([education])[0]
gender_encoded = encoders["Gender"].transform([gender])[0]
city_encoded = encoders["City"].transform([city])[0]

# PREDICTION

if st.sidebar.button("Predict Salary"):

    input_data = np.array([[
        department_encoded,
        experience,
        education_encoded,
        age,
        gender_encoded,
        city_encoded
    ]])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Monthly Salary: ₹{prediction[0]:,.2f}"
    )

# DASHBOARD METRICS

st.header("Dashboard Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Salary",
    f"₹{df['Monthly_Salary'].mean():,.0f}"
)

col2.metric(
    "Maximum Salary",
    f"₹{df['Monthly_Salary'].max():,.0f}"
)

col3.metric(
    "Total Employees",
    len(df)
)

# SALARY DISTRIBUTION

st.subheader("Salary Distribution")

fig, ax = plt.subplots(figsize=(8,5))

sns.histplot(
    df["Monthly_Salary"],
    kde=True,
    ax=ax
)

ax.set_xlabel("Monthly Salary")
ax.set_ylabel("Count")

st.pyplot(fig)

# EXPERIENCE VS SALARY

st.subheader("Experience vs Salary")

fig, ax = plt.subplots(figsize=(8,5))

sns.scatterplot(
    x=df["Experience_Years"],
    y=df["Monthly_Salary"],
    ax=ax
)

ax.set_xlabel("Experience Years")
ax.set_ylabel("Monthly Salary")

st.pyplot(fig)

# DEPARTMENT-WISE SALARY

st.subheader("Average Salary by Department")

dept_salary = df.groupby("Department")["Monthly_Salary"].mean()

fig, ax = plt.subplots(figsize=(10,5))

dept_salary.plot(kind="bar", ax=ax)

ax.set_ylabel("Average Salary")

st.pyplot(fig)

# CITY-WISE SALARY

st.subheader("City-wise Average Salary")

city_salary = df.groupby("City")["Monthly_Salary"].mean()

fig, ax = plt.subplots(figsize=(10,5))

city_salary.plot(kind="bar", ax=ax)

ax.set_ylabel("Average Salary")

st.pyplot(fig)

# GENDER SALARY DISTRIBUTION

st.subheader("Gender Salary Distribution")

fig, ax = plt.subplots(figsize=(8,5))

sns.boxplot(
    x=df["Gender"],
    y=df["Monthly_Salary"],
    ax=ax
)

st.pyplot(fig)

# CORRELATION HEATMAP

st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=['int64', 'float64'])

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# PIE CHART

st.subheader("Employee Distribution by Department")

dept_counts = df["Department"].value_counts()

fig, ax = plt.subplots(figsize=(7,7))

ax.pie(
    dept_counts,
    labels=dept_counts.index,
    autopct='%1.1f%%'
)

st.pyplot(fig)