import streamlit as st
import pandas as pd
import requests
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import requests
import streamlit_lottie as st_lottie
import PIL as image



# 🎨 Page Config
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 🛠️ Load Model + Scaler
with open("diabetespridict.sav", "rb") as f:
    saved_objects = pickle.load(f)

mod = saved_objects["model"]
scaler = saved_objects["scaler"]

# 🧮 Prediction Function
def predict(Gender, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI):
    dataaa = {
        'Gender': [Gender],
        'AGE': [AGE],
        'Urea': [Urea],
        'Cr': [Cr],
        'HbA1c': [HbA1c],
        'Chol': [Chol],
        'TG': [TG],
        'HDL': [HDL],
        'LDL': [LDL],
        'VLDL': [VLDL],
        'BMI': [BMI]
    }
    features = pd.DataFrame(dataaa)
    features_scaled = scaler.transform(features)
    prediction = mod.predict(features_scaled)
    return prediction

# 🎯 Sidebar Menu
with st.sidebar:
    choose = option_menu(
        "Navigation",
        ["Home","Visualization", "About"],
        icons=["house","bar-chart", "info-circle"],
        menu_icon="app-indicator",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "#176B87", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#176B87", "color": "white"},
        }
    )

# 🏠 Home Page
if choose == "Home":
    st.title("🩺 Diabetes Prediction App")
    st.markdown("### Enter your data to get a prediction of your health condition")

    st.markdown("#### 👤 Personal data")
    Gender = st.radio("Gender:", ("Male", "Female"))
    Gender_encoder = 1 if Gender == "Male" else 0
    AGE = st.number_input("Age:", min_value=1, max_value=120, step=1)
    Urea = float(st.text_input("Urea (mg/dL):", "0.0"))
    Cr = float(st.text_input("Creatinine (Cr) (mg/dL):", "0.0"))
    HbA1c = float(st.text_input("HbA1c (%):", "0.0"))

    Chol = float(st.text_input("Chol (mg/dL):", "0.0"))
    TG = float(st.text_input("Triglycerides (TG) (mg/dL):", "0.0"))
    HDL = float(st.text_input("Good cholesterol (HDL) (mg/dL):", "0.0"))
    LDL = float(st.text_input("Bad cholesterol (LDL) (mg/dL):", "0.0"))
    VLDL = float(st.text_input("VLDL (mg/dL):", "0.0"))

    BMI = float(st.text_input("Body mass index (BMI):", "0.0"))
    
    st.markdown("#### 🧪 Blood tests")
    Urea = st.number_input("Urea (mg/dL):", value=0.0)
    Cr = st.number_input("Creatinine (Cr) (mg/dL):", value=0.0)
    HbA1c = st.number_input("HbA1c (%):", value=0.0)

    st.markdown("#### 💉 Fats")
    Chol = float(st.text_input("Chol (mg/dL):", "0.0"))
    TG = float(st.text_input("Triglycerides (TG) (mg/dL):", "0.0"))
    HDL = float(st.text_input("Good cholesterol (HDL) (mg/dL):", "0.0"))
    LDL = float(st.text_input("Bad cholesterol (LDL) (mg/dL):", "0.0"))
    VLDL = float(st.text_input("VLDL (mg/dL):", "0.0"))

    st.markdown("#### ⚖️ Mass measures")
    BMI = float(st.text_input("Body mass index (BMI):", "0.0"))

    # Prediction
    if st.button("🔍 prediction"):
        sample = predict(Gender_encoder, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI)

        st.subheader("📊 The result:")
        if sample == 0:
            st.success("✅ The result: Healthy")
            st.balloons()
        elif sample == 1:
            st.warning("⚠️ The result:  Diabetes Exposure (Prediabetes)")
            st.snow()
        elif sample == 2:
            st.error("🚨 The result:  Diabetic")
            st.snow()


elif choose == "About":
    st.title("ℹ️ About This App")

    st.subheader("🎯 Purpose")
    st.markdown("""
    This application is designed to **predict diabetes status** (Healthy, Prediabetic, Diabetic) 
    using clinical health data and a machine learning model.  
    The goal is to **raise awareness** about diabetes risk and provide a practical tool 
    that can support both **medical professionals** and the **general public**.
    """)

    st.subheader("📊 Dataset")
    st.markdown("""
    - **Source:** Collected from a local hospital's patient records.  
    - **Size:** 265 patient records, 12 features  
    - **Preprocessing:** Data cleaning and normalization were applied to ensure reliable input for the model.  
    """)

    st.subheader("🤖 Machine Learning Model")
    st.markdown("""
    - **Model Used:** Random Forest Classifier  
    - **Why Random Forest?**  
      - Handles small datasets (265 rows) effectively.  
      - Robust against overfitting compared to single decision trees.  
      - Works well with mixed data distributions (continuous + categorical).  
      - Provides high accuracy and interpretability in medical applications.  

    **Model Performance (Test Set):**
    - Precision: 0.95 – 1.00 – 1.00
    - Recall: 1.00 – 1.00 – 0.96  
    - F1-score: 0.97 – 1.00 – 0.98  
    - **Accuracy:** 98%  
    """)

    st.subheader("📌 Variables Used")
    st.markdown("""
    - 👤 **Gender**: (0 = Female, 1 = Male). May influence risk due to biological differences.  
    - 📅 **Age (AGE)**: Older individuals generally have higher risk of diabetes.  
    - 🧪 **Urea**: High levels may indicate kidney dysfunction linked to diabetes complications.  
    - 🧪 **Creatinine (Cr)**: Marker of kidney function. Elevated in diabetic kidney disease.  
    - 🧪 **HbA1c**: Average blood glucose over 2–3 months.  
        - Normal < 5.7%, Prediabetic 5.7–6.4%, Diabetic ≥ 6.5%.  
    - 💉 **Cholesterol (Chol)**: Total cholesterol in the blood. High levels raise cardiovascular risk.  
    - 💉 **Triglycerides (TG)**: High TG is associated with insulin resistance.  
    - 💉 **HDL ("Good" Cholesterol)**: Low levels increase diabetes and heart disease risk.  
    - 💉 **LDL ("Bad" Cholesterol)**: High levels worsen cardiovascular complications in diabetes.  
    - 💉 **VLDL**: Another form of "bad" cholesterol carrying triglycerides.  
    - ⚖️ **BMI (Body Mass Index)**: Overweight (≥25) and obesity (≥30) strongly increase diabetes risk.  
    """)

    st.subheader("⚠️ Disclaimer")
    st.info("""
    This application is for **educational and awareness purposes** only.  
    It does not replace professional medical consultation or diagnosis.  
    Always consult a healthcare provider for medical advice.  
    """)
# 📊 Visualization Page
if choose == "Visualization":
    st.title("📊 Data Visualization")
    st.markdown("###   Visual Data Exploration")
    file =st.file_uploader("Upload file", type=["csv"])
    if file is not None:
        data = pd.read_csv(file)
        n_rows=st.slider("Number of rows to display  ", min_value=5, max_value=len(data), value=5, step=1)
        columns_to_show= st.multiselect("Select columns to display", options=data.columns.tolist(), default=data.columns.tolist())
        st.dataframe(data[:n_rows][columns_to_show])

        st.markdown("#### Distribution of Target Variable (Class)")
        st.bar_chart(data['Class'].value_counts())

        st.markdown("#### Statistical Summary")
        st.write(data.describe())

       
       
        tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Histogram", "Box Plot"])

        
        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_column = st.selectbox("Select X-axis column", options=data.select_dtypes(include=np.number).columns.tolist())
            with col2:
                y_column = st.selectbox("Select Y-axis column", options=data.select_dtypes(include=np.number).columns.tolist())
            with col3:
                color_column = st.selectbox("Select column for color", data.columns)
            
            scatter_fig = px.scatter(
                data,
                x=x_column,
                y=y_column,
                color=color_column,
                title=f'{x_column} vs {y_column} Scatter Plot'
            )
            st.plotly_chart(scatter_fig)

        
        with tab2:
            hist_column = st.selectbox(
                "Select column for histogram",
                options=data.select_dtypes(include=np.number).columns.tolist(),
                key='hist'
            )
            hist_fig = px.histogram(
                data,
                x=hist_column,
                nbins=30,
                title=f'{hist_column} Histogram'
            )
            st.plotly_chart(hist_fig)

        
        with tab3:
            box_column = st.selectbox(
                "Select column for box plot",
                options=data.select_dtypes(include=np.number).columns.tolist(),
                key='box'
            )
            class_column = st.selectbox(
                "Select class column",
                options=data.columns,
                key='class'
            )
            
            box_fig = px.box(
                data,
                x=class_column,
                y=box_column,
                color=class_column,
                title=f'{box_column} Distribution by {class_column}'
            )
            st.plotly_chart(box_fig)

