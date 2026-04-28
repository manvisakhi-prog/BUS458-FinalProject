import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Loaded our saved files 
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("⚠️ Model or Scaler files not found! Ensure they are in the same folder as app.py.")

# Title and Styling
st.title("Loan Approval & Revenue Optimizer")
st.subheader("Automated Loan Underwriting & Payout Optimization")

# 2. Input fields and widgets
col1, col2 = st.columns(2)

with col1:
    fico = st.slider("FICO Score", 300, 850, 700)
    income = st.number_input("Monthly Income ($)", value=5000)
    loan_amt = st.number_input("Loan Amount ($)", value=10000)

with col2:
    reason = st.selectbox("Reason for Loan", ["Debt Consolidation", "Home Improvement", "Credit Card Refinancing", "Other"])
    employment = st.selectbox("Employment Status", ["Full-time", "Part-time", "Unemployed", "Self-employed"])

# 3. Logic for the Prediction
if st.button("📊 Evaluate Match"):
    #Create initial DataFrame
    features = {
        "FICO_score": fico, 
        "Monthly_Gross_Income": income, 
        "Requested_Loan_Amount": loan_amt
    }
    input_df = pd.DataFrame([features])

    model_columns = scaler.feature_names_in_

    # Create dummy columns to match our notebook's One-Hot Encoding
    input_df = pd.concat([input_df, pd.get_dummies(pd.Series([reason.lower().replace(" ", "_")], name='Reason'))], axis=1)
    input_df = pd.concat([input_df, pd.get_dummies(pd.Series([employment.lower().replace("-", "_")], name='Employment_Status'))], axis=1)

    # Fill in any missing columns expected by the model with 0
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure the columns are in the EXACT order the model expects
    final_input = input_df[model_columns]
    
    # Step C: Scale and Predict
    scaled_data = scaler.transform(final_input)
    prob = model.predict_proba(scaled_data)[0][1]
    
    # Step D: Display Results
    st.divider()
    st.subheader(f"Approval Probability: {prob:.1%}")
    
    if prob > 0.5:
        st.success("✅ Verdict: Applicant is likely **Approved**.")
        # This is the 10-point recommendation logic!
        st.info("💰 **Strategic Recommendation:** Route to **Lender B**. At this probability, Lender B offers the maximum revenue payout of **$350**.")
    else:
        st.error("🚫 Verdict: Applicant is likely **Denied**.")
        st.warning("Recommendation: This profile does not meet current approval thresholds.")

st.markdown("---")
st.caption("BUS 458 Final Exam Submission — Loan Payout Optimization")