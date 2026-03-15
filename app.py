import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ================================
# Train model in background
# ================================
@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
    df = pd.read_csv(url)
    
    # Drop useless columns
    df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
    
    # Encode categorical columns
    le_dict = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, 
                                    class_weight='balanced',
                                    random_state=42)
    model.fit(X_train, y_train)
    
    return model, le_dict, X.columns.tolist()

# ================================
# App UI
# ================================
st.title("👥 HR Employee Attrition Predictor")
st.write("Predict whether an employee is likely to leave the company!")

# Load model
model, le_dict, feature_names = train_model()

st.subheader("Employee Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 60, 30)
    monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4],
                                     help="1=Low, 4=High")

with col2:
    distance = st.slider("Distance From Home (km)", 1, 30, 5)
    years_company = st.slider("Years At Company", 0, 40, 3)
    work_life = st.selectbox("Work Life Balance", [1, 2, 3, 4],
                              help="1=Low, 4=High")
    environment = st.selectbox("Environment Satisfaction", [1, 2, 3, 4],
                                help="1=Low, 4=High")

with col3:
    department = st.selectbox("Department", 
                               ["Sales", "Research & Development", "Human Resources"])
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    marital_status = st.selectbox("Marital Status", 
                                   ["Single", "Married", "Divorced"])
    stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])

# Predict button
if st.button("🔮 Predict Attrition", type="primary"):
    
    # Create input dataframe with all required features
    input_dict = {
        'Age': age,
        'BusinessTravel': 1,
        'DailyRate': 800,
        'Department': le_dict['Department'].transform([department])[0],
        'DistanceFromHome': distance,
        'Education': 3,
        'EducationField': 1,
        'EnvironmentSatisfaction': environment,
        'Gender': 1,
        'HourlyRate': 66,
        'JobInvolvement': 3,
        'JobLevel': job_level,
        'JobRole': 1,
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': le_dict['MaritalStatus'].transform([marital_status])[0],
        'MonthlyIncome': monthly_income,
        'MonthlyRate': 14000,
        'NumCompaniesWorked': 2,
        'OverTime': le_dict['OverTime'].transform([overtime])[0],
        'PercentSalaryHike': 14,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'StockOptionLevel': stock_option,
        'TotalWorkingYears': years_company + 2,
        'TrainingTimesLastYear': 3,
        'WorkLifeBalance': work_life,
        'YearsAtCompany': years_company,
        'YearsInCurrentRole': 2,
        'YearsSinceLastPromotion': 1,
        'YearsWithCurrManager': 2
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Predict
    probability = model.predict_proba(input_df)[0]
    attrition_prob = probability[1]
    
    st.divider()
    
    if attrition_prob >= 0.3:
        st.error(f"⚠️ High Attrition Risk! Probability: {attrition_prob*100:.1f}%")
        st.write("**Recommendations:**")
        if overtime == "Yes":
            st.write("- 🕐 Reduce overtime hours")
        if job_satisfaction <= 2:
            st.write("- 😊 Improve job satisfaction")
        if monthly_income < 5000:
            st.write("- 💰 Consider salary increase")
        if distance > 20:
            st.write("- 🏠 Consider remote work options")
    else:
        st.success(f"✅ Low Attrition Risk! Probability: {attrition_prob*100:.1f}%")
        st.balloons()
    
    # Risk meter
    st.subheader("Risk Level")
    st.progress(attrition_prob)

