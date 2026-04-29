import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

# 1. Load the model
model = pickle.load(open("my_model.pkl", "rb"))

# we wanted color to be black,white,red (help from gemini a bit) 
st.markdown("""
    <style>
    /* 1. Background */
    .stApp { 
        background-color: #FFFFFF !important; }
    
    /* 2. Main Title (Red) */
    h1 { 
        color: #CC0000 !important; 
        text-align: center; 
        border-bottom: 3px solid #000000; }

    /* 3. THE FIX: Probability Header & Subheaders (Black) */
    h3, .stSubheader, [data-testid="stHeader"] {
        color: #000000 !important;}

    /* 4. All Labels, Slider Numbers, and Radio Text (Black) */
    label, .stWidgetLabel, p, 
    [data-baseweb="typo-label-small"], 
    div[data-testid="stMarkdownContainer"] p,
    span[data-baseweb="typo-label-small"] {
        color: #000000 !important;
        font-weight: bold !important;}

    /* 5. Results Probability Text specifically */
    .stMarkdown h3 {
        color: #000000 !important; }

    /* 6. Red Button */
    div.stButton > button:first-child {
        background-color: #CC0000;
        color: white !important;
        border: 2px solid #000000;
        width: 100%;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏦 WOLF FINANCIAL UNDERWRITING PORTAL")

#Input Fields
col1, col2 = st.columns(2)

with col1:
    fico = st.slider("FICO Score", 300, 850, 700)
    income = st.number_input("Monthly Gross Income ($)", value=5000)
    loan_amt = st.number_input("Requested Loan Amount ($)", value=10000)
    housing = st.number_input("Monthly Housing Payment ($)", value=1200)

with col2:
    reason = st.selectbox("Reason for Loan", ["Credit Card Refinancing", "Debt Conslidated", "Home Improvement", "Major Purchase", "Other"])
    employment = st.selectbox("Employment Status", ["Full-time", "Part-time", "Unemployed"])
    
    # Yes/No Picker 
    bankrupt_choice = st.radio("Ever Bankrupt or Foreclosed?", ["No", "Yes"])
    bankrupt = 1 if bankrupt_choice == "Yes" else 0

#Prediction Logic
if st.button("RUN ANALYSIS"):
    # Build the input data
    features = {
        "FICO_score": fico, 
        "Monthly_Gross_Income": income, 
        "Requested_Loan_Amount": loan_amt,
        "Granted_Loan_Amount": loan_amt, 
        "Monthly_Housing_Payment": housing,
        "Ever_Bankrupt_or_Foreclose": bankrupt
    }
    input_df = pd.DataFrame([features])
    model_columns = model.feature_names_in_


    reason_col = f"Reason_{reason.lower().replace(' ', '_')}"
    input_df[reason_col] = 1
    
    emp_col = f"Employment_Status_{employment.lower().replace('-', '_')}"
    input_df[emp_col] = 1

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    final_input = input_df[model_columns]
    prob = model.predict_proba(final_input)[0][1]
    
    #Results for the different lenders
    st.divider()
    st.subheader(f"Probability of Approval: {prob:.1%}")

    if prob > 0.70:
        st.success("✅ **High Confidence Approval**")
        st.info(f"💰 **Strategy:** Route to **Lender B**. Applicant qualifies for our premium partner. **Estimated Payout: $350**")
        
    elif prob >= 0.50:
        st.success("✅ **Standard Approval**")
        st.info(f"💰 **Strategy:** Route to **Lender A**. Solid applicant, but may not meet Lender B's strict tier. **Estimated Payout: $250**")
        
    elif prob > 0.35:
        st.warning("⚠️ **Marginal Approval**")
        st.info(f"💰 **Strategy:** Route to **Lender C**. Higher likelihood of closing with this flexible partner. **Estimated Payout: $150**")
        
    else:
        st.error("🚫 **Application Denied**")
        st.write("The risk profile exceeds the tolerance of our current lending partners (A, B, or C).")

    # Your contact footer
    st.markdown("<small>Questions? Contact msharm25@ncsu.edu or pjshah3@ncsu.edu</small>", unsafe_allow_html=True)
