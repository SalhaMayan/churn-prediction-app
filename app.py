
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("RandomForstchurn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #efefee;
        color: #333;
    }
    .stButton>button {
        background-color: #467267;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #3b5c52;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h2 style='text-align: left;'>Customer Churn Prediction</h2>", unsafe_allow_html=True)
st.markdown("This application predicts customer churn using a Random Forest Classifier, based on customer profile and behavior.")
st.markdown("---")
st.markdown("### Customer Profile")

col1, col2 = st.columns(2)

with col1:
    SatisfactionScore = st.selectbox("Customer Satisfaction (1 = Low, 5 = High)", [1, 2, 3, 4, 5], index=2)
    CashbackAmount = st.number_input("Total Cashback Received (SAR)", min_value=0.0)
    DaySinceLastOrder = st.number_input("Days Since Last Purchase", min_value=0)
    category = st.selectbox("Most Frequent Purchase Category", ["Mobile", "Laptop & Accessory", "Other"])
    PreferedOrderCat_Mobile = 1 if category == "Mobile" else 0
    PreferedOrderCat_Laptop = 1 if category == "Laptop & Accessory" else 0

with col2:
    Complain_input = st.radio("Has the customer made a complaint?", ["Yes", "No"])
    Complain = 1 if Complain_input == "Yes" else 0
    Tenure = st.number_input("Tenure with Company (Months)", min_value=0)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    MaritalStatus_Single = 1 if marital_status == "Single" else 0
    MaritalStatus_Married = 1 if marital_status == "Married" else 0
    MaritalStatus_Divorced = 1 if marital_status == "Divorced" else 0
    NumberOfDeviceRegistered = st.selectbox("Number of Devices Used by Customer", list(range(1, 11)), index=0)

st.markdown("---")

if st.button("Predict"):
    input_data = pd.DataFrame([[
        Complain,
        PreferedOrderCat_Mobile,
        MaritalStatus_Single,
        NumberOfDeviceRegistered,
        SatisfactionScore,
        CashbackAmount,
        MaritalStatus_Married,
        MaritalStatus_Divorced,
        DaySinceLastOrder,
        PreferedOrderCat_Laptop,
        Tenure
    ]], columns=[
        'Complain',
        'PreferedOrderCat_Mobile',
        'MaritalStatus_Single',
        'NumberOfDeviceRegistered',
        'SatisfactionScore',
        'CashbackAmount',
        'MaritalStatus_Married',
        'MaritalStatus_Divorced',
        'DaySinceLastOrder',
        'PreferedOrderCat_Laptop & Accessory',
        'Tenure'
    ])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.markdown("### Prediction Result:")
    if prediction[0] == 1:
        st.markdown("<h3 style='color:red;'>The customer is likely to churn.</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:green;'>The customer is likely to stay.</h3>", unsafe_allow_html=True)

    readable_data = {
        "Complaint": Complain_input,
        "Preferred Category": category,
        "Marital Status": marital_status,
        "Devices Registered": NumberOfDeviceRegistered,
        "Satisfaction Score": SatisfactionScore,
        "Cashback Amount (SAR)": CashbackAmount,
        "Tenure (Months)": Tenure,
        "Days Since Last Order": DaySinceLastOrder
    }

    st.markdown("### Customer Info Summary")
    st.table(pd.DataFrame([readable_data]))
