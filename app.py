import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

# 1. Load your specific files
try:
    # UPDATED: Matches the filename 'my_model.pkl' you provided 
    model = pickle.load(open("my_model.pkl", "rb"))
except FileNotFoundError:
    st.error("⚠️ Files not found! Ensure 'my_model.pkl' and 'scaler.pkl' are in the same folder as app.py.")

# We wanted our colors to be black,white,and red
# --- THEME CUSTOMIZATION (Red, White, Black) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    h1 {
        color: #CC0000 !important;
        text-align: center;
        border-bottom: 3px solid #000000;
        padding-bottom: 10px;
    }
    div.stButton > button:first-child {
        background-color: #CC0000;
        color: white;
        border: 2px solid #000000;
        width: 100%;
    }

    /* Fix for slider min/max (300/850) */
    .stSlider [data-baseweb="typo-label-small"] {
        color: #000000 !important;
        font-weight: bold !important;
    }

    /* Fix for slider value bubble */
    div[data-testid="stThumbValue"] {
        color: #000000 !important;
        font-weight: bold !important;
    }

    /* NEW: Fix for Radio button labels (Yes/No or 0/1) */
    div[data-testid="stRadio"] label p {
        color: #000000 !important;
        font-weight: bold !important;
    }

    /* Fix for all other general labels and text */
    label, .stWidgetLabel, .stMarkdown p {
        color: #000000 !important;
        font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏦 LOAN APPROVAL OPTIMIZER")
st.subheader("Revenue Maximization & Payout Logic")

# 2. Input fields (Grouping into columns for a cleaner look)
st.markdown("### **Applicant Data Entry**")
col1, col2 = st.columns(2)

with col1:
    fico = st.slider("FICO Score", 300, 850, 700)
    income = st.number_input("Monthly Gross Income ($)", value=5000)
    loan_amt = st.number_input("Requested Loan Amount ($)", value=10000)
    housing = st.number_input("Monthly Housing Payment ($)", value=1200)

with col2:
    reason = st.selectbox("Reason for Loan", ["Credit Card Refinancing", "Debt Consolidation", "Home Improvement", "Major Purchase", "Other"])
    employment = st.selectbox("Employment Status", ["Full-time", "Part-time", "Unemployed"])
    bankrupt = st.radio("Ever Bankrupt or Foreclosed?", [0, 1], help="0 = No, 1 = Yes")

# 3. Prediction Logic
if st.button("RUN ANALYSIS"):
    # Create feature set based on your model's requirement 
    features = {
        "FICO_score": fico, 
        "Monthly_Gross_Income": income, 
        "Requested_Loan_Amount": loan_amt,
        "Granted_Loan_Amount": loan_amt,
        "Monthly_Housing_Payment": housing,
        "Ever_Bankrupt_or_Foreclose": bankrupt
    }
    input_df = pd.DataFrame([features])

    model_columns = scaler.feature_names_in_

    # Encoding categories to match 'my_model.pkl' labels 
    reason_col = f"Reason_{reason.lower().replace(' ', '_')}"
    input_df[reason_col] = 1
    
    emp_col = f"Employment_Status_{employment.lower().replace('-', '_')}"
    input_df[emp_col] = 1

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    final_input = input_df[model_columns]
    
    # Scale and Predict
    scaled_data = scaler.transform(final_input)
    prob = model.predict_proba(scaled_data)[0][1]
    
    # 4. Display Results in themed boxes
    st.markdown("---")
    if prob > 0.5:
        st.success(f"### Probability of Approval: {prob:.1%}")
        st.balloons()
        # Strategy Recommendation to meet objective 
        st.info("🎯 **LENDER STRATEGY:** Route to **Lender B**. Maximum Payout: **$350**.")
    else:
        st.error(f"### Probability of Approval: {prob:.1%}")
        st.warning("⚠️ **LENDER STRATEGY:** Reject or Request Additional Documentation.")

st.markdown("<br><hr><center><small>BUS 458 Final Exam | Red-Black Edition</small></center>", unsafe_allow_html=True)
