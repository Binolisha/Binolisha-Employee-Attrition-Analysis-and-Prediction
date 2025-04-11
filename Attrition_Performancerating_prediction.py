import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import os

# Set wide layout
st.set_page_config(layout="wide")

# Load dataset
@st.cache_data
def load_data():
    # Use read_excel for .xlsx files
    return pd.read_excel(r"C:\Users\DELL\Downloads\Employee-Attrition.xlsx")

df = load_data()

# Define paths to model files
model_paths = {
    "Attrition Prediction":r"C:\Users\DELL\attrition_prediction_model.pkl",
    "Performance Rating Prediction":r"C:\Users\DELL\performancerating_prediction_model.pkl"
}
#Define function to load the model file
def load_model(model_type):
    model_path = model_paths.get(model_type)
    if model_path and os.path.exists(model_path):
        try:
            return joblib.load(model_path)
            if isinstance(loaded, tuple):
                model, reverse_map = loaded
                return model, reverse_map
            else:
                return loaded, None
        except Exception as e:
            st.write(f"{e} in loading the file ")
            return None,None
    else:
        st.write(f"The file is not found:{model_path}")
    return None,None

# === Sidebar ===
with st.sidebar:
    st.markdown("## EMPLOYEE ANALYSIS AND PREDICTION")
    menu = {
        "Overview": "🏠 Overview",
        "Performance Insights": "📈 Performance Insights",
        "Attrition Analysis": "🔍 Attrition Analysis",
        "Predictive Modeling": "📉 Predictive Modeling"
    }
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    for page_key, page_label in menu.items():
        if st.button(page_label, use_container_width=True):
            st.session_state.page = page_key

st.title(st.session_state.page)

# === Page: Overview ===
if st.session_state.page == "Overview":
    tab_option = st.selectbox("Select a tab to see details:", ["Dataset Overview", "Descriptive Statistics", "Exploratory Data Analysis"])
    st.markdown("""
    Employee turnover poses a significant challenge for organizations...
    """)

    if tab_option == "Dataset Overview":
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head())
        st.markdown("### 🔑 Column Names")
        st.table(pd.DataFrame(df.columns,columns=["Column Names"]))
        st.markdown("### 📊 Dataset Shape")
        st.write(f"{df.shape[0]} rows, {df.shape[1]} columns")

    elif tab_option == "Descriptive Statistics":
        st.subheader("📈 Numeric Columns Statistics")
        st.write(df.describe())
        st.subheader("📈 Categorical Columns Statistics")
        st.write(df.describe(include=["object"]))

    elif tab_option == "Exploratory Data Analysis":
        st.subheader("📊 Basic EDA")
        categorical_columns = df.select_dtypes(include='object').columns.tolist()
        selected_column = st.selectbox("Select a column to visualize", categorical_columns)
        if selected_column:
            fig = px.histogram(df, x=selected_column, color=selected_column)
            st.plotly_chart(fig, use_container_width=True)

# === Page: Performance Insights ===
elif st.session_state.page == "Performance Insights":
    st.markdown("Here you'll see employee performance analytics.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Performance Rating", round(df["PerformanceRating"].mean(), 2))
    with col2:
        st.metric("Avg Job Satisfaction", round(df["JobSatisfaction"].mean(), 2))
    with col3:
        st.metric("Avg Work-Life Balance", round(df["WorkLifeBalance"].mean(), 2))

    chart_options = [
        "Correlation Heatmap",
        "Performance Rating vs Monthly Income Distribution (Boxplot)",
        "Monthly Income Distribution by Performance Rating"
    ]
    selected_chart = st.selectbox("Select a chart to view", chart_options)
    if selected_chart == "Correlation Heatmap":
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(12, 8))  # Bigger canvas
        sns.heatmap(corr,annot=True,cmap="coolwarm",fmt=".2f",annot_kws={"size": 10})
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
    elif selected_chart == "Performance Rating vs Monthly Income Distribution (Boxplot)":
        fig = px.box(df, x="PerformanceRating", y="MonthlyIncome", color="PerformanceRating")
        st.plotly_chart(fig)
    elif selected_chart == "Monthly Income Distribution by Performance Rating":
        fig = px.histogram(df, x="MonthlyIncome", color="PerformanceRating", barmode="overlay",color_discrete_map=
        {"4": "#FF6B6B","5": "#4D96FF" })
        st.plotly_chart(fig)

# === Page: Attrition Analysis ===
elif st.session_state.page == "Attrition Analysis":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ATTRITION COUNT (Yes)", df[df["Attrition"] == "Yes"].shape[0])
    with col2:
        st.metric("ATTRITION COUNT (No)", df[df["Attrition"] == "No"].shape[0])
    with col3:
        st.metric("ATTRITION RATE (%)", round((df["Attrition"] == "Yes").mean() * 100, 2))

    chart_options = [
        "Attrition by Jobrole", "Attrition Breakdown", "Attrition by Department",
        "Attrition by Gender", "Attrition by MaritalStatus",
        "Age distribution by employeeLeaving", "Attrition Vs Monthly income",
        "Attrition Vs Work-Life Balance"
    ]
    selected_chart = st.selectbox("Select a chart to view", chart_options)

    if selected_chart == "Attrition by Jobrole":
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="JobRole", hue="Attrition")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    elif selected_chart == "Attrition Breakdown":
        attrition_counts = df["Attrition"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title("Attrition Breakdown")
        st.pyplot(fig)
    elif selected_chart == "Attrition by Department":
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Department", hue="Attrition")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    elif selected_chart == "Attrition by Gender":
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Gender", hue="Attrition")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    elif selected_chart == "Attrition by MaritalStatus":
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="MaritalStatus", hue="Attrition")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    elif selected_chart == "Attrition by MaritalStatus":
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="MaritalStatus", hue="Attrition")
        plt.xticks(rotation=45)
        st.pyplot(fig) 
    elif selected_chart == "Age distribution by employeeLeaving":
        fig = px.histogram(df,x="Age",color="Attrition",barmode="overlay",nbins=30,color_discrete_map={
        "Yes": "#FF6B6B","No": "#4D96FF" })
        st.plotly_chart(fig)
    elif selected_chart == "Attrition Vs Monthly income":
        fig = px.box(df,x="Attrition",y="MonthlyIncome",color="Attrition",color_discrete_map={
        "Yes": "#FF6B6B","No": "#4D96FF"},points="all" )
        st.plotly_chart(fig)

    elif selected_chart == "Attrition Vs Work-Life Balance":
        fig = px.histogram(df,x="WorkLifeBalance",color="Attrition",barmode="group",color_discrete_map={
        "Yes": "#FF6B6B","No": "#4D96FF"})
        st.plotly_chart(fig)
    
# --- PREDICTIVE MODELING ---
elif st.session_state.page == "Predictive Modeling":
    st.subheader("Choose a Prediction Task")
    prediction_task = st.radio("Select Task", list(model_paths.keys()))
    businesstravel_map={"Travel Rarely":2,"Travel Frequently":1,"Non-Travel":0}
    department_map={'Sales':2, 'Research & Development':1, 'Human Resources':0}
    EducationField_map={'Life Sciences':1, 'Other':4, 'Medical':3, 'Marketing':2,'Technical Degree':5, 'Human Resources':0}
    gender_map = {'Male': 0, 'Female': 1}
    JobRole_map={'Sales Executive':7, 'Research Scientist':6, 'Laboratory Technician':2,'Manufacturing Director':4, 'Healthcare Representative':0, 'Manager':3,'Sales Representative':8, 'Research Director':5, 'Human Resources':1}
    MartialStatus_map={'Single':2, 'Married':1, 'Divorced':0}
    OverTime_map={"Yes":1,"No":0}
    if prediction_task == "Attrition Prediction":
        st.subheader(f"Enter the Below details to make Attrition Prediction!")
        
        def get_user_input_attrition():
            return np.array([[
                st.number_input("Enter Your Age:",min_value=0,max_value=100,step=1),
                businesstravel_map[st.radio("Traveling for Business",["Travel Rarely","Travel Frequently","Non-Travel"])],
                department_map[st.radio("Department",['Sales', 'Research & Development', 'Human Resources'])],
                st.number_input("Enter Distance from your Home:",min_value=0,max_value=30,step=1),
                st.slider("Enter your Education Level",min_value=1,max_value=5,step=1),
                EducationField_map[st.selectbox("Enter your Education field",['Life Sciences', 'Other', 'Medical', 'Marketing','Technical Degree', 'Human Resources'])],
                st.radio("Environmental Satisfaction Level",[1,2,3,4]),
                gender_map[st.selectbox('Gender', ['Male', 'Female'])],
                st.radio("Job Involvement Rating",[1,2,3,4]),
                JobRole_map[st.selectbox("Enter your Job Role",['Sales Executive', 'Research Scientist','LaboratoryTechnician', 'Manufacturing Director', 'Healthcare Representative', 'Manager','Sales Representative', 'Research Director', 'Human Resources'])],
                st.radio("Job Statisfaction Rating",[1,2,3,4]),
                st.radio("Job Level",[1,2,3,4,5]),
                MartialStatus_map[st.radio("Enter Your Marital Status",['Single', 'Married', 'Divorced'])],
                st.number_input("Enter your monthly Salary",min_value=0.0,max_value=20000.0,step=10.0),
                st.slider("Enter Number of Companies you Worked:",min_value=0,max_value=10,step=1),
                OverTime_map[st.radio("Do you work Over Time?",["Yes","No"])],
                st.slider("Enter the Percentage of Salary Hike you get:",min_value=0,max_value=50,step=1),
                st.radio("Performance Rating",[3,4]),
                st.number_input("Total Working Years in Company",min_value=0,max_value=50,step=1)
            ]])
        user_input = get_user_input_attrition()
        if st.button("Predict Attrition"):
            model = load_model("Attrition Prediction")
            if model:
                try:
                    prediction = int(model.predict(user_input).item())
                    if prediction == 1:
                        st.success("✅ Employee likely to leave the company.")
                    else:
                        st.success("❌ Employee unlikely to leave the company.")
                except Exception as e:
                    st.error(f"⚠️ Prediction failed: {e}")
            else:
                st.error("⚠️ No model available for Attrition prediction.")

    # Prediction 2: Performance Rating
    elif prediction_task == "Performance Rating Prediction":
        st.subheader(f"Enter the Below details to make Performance Rating Prediction!")
        
        def get_user_input_performance(): 
            return np.array([[
                st.number_input("Enter Your Age:",min_value=0,max_value=100,step=1),
                businesstravel_map[st.radio("Traveling for Business",["Travel Rarely","Travel Frequently","Non-Travel"])],
                department_map[st.radio("Department",['Sales', 'Research & Development', 'Human Resources'])],
                st.number_input("Enter Distance from your Home:",min_value=0,max_value=30,step=1),
                st.radio("Environmental Satisfaction Level",[1,2,3,4]),
                gender_map[st.selectbox('Gender', ['Male', 'Female'])],
                JobRole_map[st.selectbox("Enter your Job Role",['Sales Executive', 'Research Scientist','LaboratoryTechnician', 'Manufacturing Director', 'Healthcare Representative', 'Manager','Sales Representative', 'Research Director', 'Human Resources'])],
                st.radio("Job Statisfaction Rating",[1,2,3,4]),
                st.number_input("Enter your monthly Salary",min_value=0.0,max_value=20000.0,step=10.0),
                OverTime_map[st.radio("Do you work Over Time?",["Yes","No"])],
                st.slider("Enter the Percentage of Salary Hike you get:",min_value=0,max_value=50,step=1),
                st.number_input("Total Working Years in Company",min_value=0,max_value=50,step=1),
                st.slider("Enter the training times for the last year:",min_value=0,max_value=6,step=1),
                st.radio("Work Life Balance Rating",[1,2,3,4]),           
                st.slider("Enter the number of years worked in the company:",min_value=0,max_value=40,step=1),
                st.slider("Enter the number of years worked for current role in the company:",min_value=0,max_value=40,step=1),
                st.slider("Enter the number of years since last promotion in the company:",min_value=0,max_value=15,step=1),
                st.slider("Enter the number of years with current manager in the company:",min_value=0,max_value=15,step=1)               ]])
            
        user_input = get_user_input_performance()
        if st.button("Predict Performance Rating"):
            model, reverse_map = load_model("Performance Rating Prediction")
            if model:
                try:
                    prediction = int(model.predict(user_input).item())
                    if reverse_map:
                        prediction = reverse_map[prediction]  # convert back to original label (3 or 4)
                    st.success(f"📈 Predicted Performance Rating: {prediction}")
                except Exception as e:
                    st.error(f"⚠️ Prediction failed: {e}")
            else:
                st.error("⚠️ No model available for Performance prediction.")
        
