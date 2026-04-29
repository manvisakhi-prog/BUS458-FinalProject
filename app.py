import streamlit as st
import pandas as pd
import pickle
import numpy as np

model = pickle.load(open("my_model.pkl", "rb")) 
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title and Styling
st.title("Loan Approval & Revenue Optimizer")
st.subheader("Automated Loan Underwriting & Payout Optimization")

# 2. Input fields and widgets
st.header("Applicant Details")
col1, col2 = st.columns(2)

with col1:
    fico = st.slider("FICO Score", 300, 850, 700)
    income = st.number_input("Monthly Gross Income ($)", value=5000)
    loan_amt = st.number_input("Requested Loan Amount ($)", value=10000)
    # NEW: Your model requires housing payment info 
    housing = st.number_input("Monthly Housing Payment ($)", value=1200)

with col2:
    reason = st.selectbox("Reason for Loan", ["Credit Card Refinancing", "Debt Consolidation", "Home Improvement", "Major Purchase", "Other"])
    employment = st.selectbox("Employment Status", ["Full-time", "Part-time", "Unemployed"])
    # NEW: Your model requires bankruptcy history 
    bankrupt = st.radio("Ever Bankrupt or Foreclosed?", [0, 1], help="0 = No, 1 = Yes")

# 3. Logic for the Prediction
if st.button("📊 Evaluate Match"):
    # Create initial DataFrame with ALL numeric fields your model expects 
    features = {
        "FICO_score": fico, 
        "Monthly_Gross_Income": income, 
        "Requested_Loan_Amount": loan_amt,
        "Granted_Loan_Amount": loan_amt, # Assuming granted matches requested for the simulation
        "Monthly_Housing_Payment": housing,
        "Ever_Bankrupt_or_Foreclose": bankrupt
    }
    input_df = pd.DataFrame([features])

    # Get the exact 29 columns your model expects 
    model_columns = scaler.feature_names_in_

    # Create dummy columns to match your model's specific One-Hot Encoding names 
    # This matches names like 'Reason_debt_conslidation' found in your pickle file 
    reason_col = f"Reason_{reason.lower().replace(' ', '_')}"
    input_df[reason_col] = 1
    
    emp_col = f"Employment_Status_{employment.lower().replace('-', '_')}"
    input_df[emp_col] = 1

    # Fill in all other 29 columns (Sectors, Fico Groups, etc.) with 0 
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure the columns are in the EXACT order the model expects 
    final_input = input_df[model_columns]
    
    # Scale and Predict
    scaled_data = scaler.transform(final_input)
    prob = model.predict_proba(scaled_data)[0][1]
    
    # Display Results
    st.divider()
    st.subheader(f"Approval Probability: {prob:.1%}")
    
    if prob > 0.5:
        st.success("✅ Verdict: Applicant is likely **Approved**.")
        # This recommendation satisfies the revenue maximization requirement 
        st.info("💰 **Strategic Recommendation:** Route to **Lender B**. At this probability, Lender B offers the maximum revenue payout of **$350**.")
    else:
        st.error("🚫 Verdict: Applicant is likely **Denied**.")
        st.warning("Recommendation: This profile does not meet current approval thresholds.")

st.markdown("---")
st.caption("BUS 458 Final Exam Submission — Loan Payout Optimization")
