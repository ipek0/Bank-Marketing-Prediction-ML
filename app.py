import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stButton>button {
        width: 100%;
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stButton>button:hover {background-color: #27ae60;}
    </style>
""", unsafe_allow_html=True)

# Constants
MODEL_PATH = 'final_bank_marketing_model.joblib'

@st.cache_resource
def load_model():
    """Load the trained pipeline"""
    if not os.path.exists(MODEL_PATH):
        st.error(f" Model file not found: {MODEL_PATH}")
        st.info("Please run the Jupyter notebook first to train and save the model.")
        return None
    
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_user_input():
    """Collect user input from sidebar"""
    st.sidebar.header(" Client Information")
    
    # Personal Information
    st.sidebar.subheader("Personal Details")
    age = st.sidebar.slider("Age", 18, 95, 35)
    
    job = st.sidebar.selectbox("Job", [
        'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
        'management', 'retired', 'self-employed', 'services', 
        'student', 'technician', 'unemployed', 'unknown'
    ])
    
    marital = st.sidebar.selectbox("Marital Status", 
        ['married', 'single', 'divorced', 'unknown'])
    
    education = st.sidebar.selectbox("Education", [
        'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
        'illiterate', 'professional.course', 'university.degree', 'unknown'
    ])
    
    # Financial Information
    st.sidebar.subheader("Financial Status")
    default = st.sidebar.selectbox("Credit in Default?", ['no', 'yes', 'unknown'])
    housing = st.sidebar.selectbox("Housing Loan?", ['no', 'yes', 'unknown'])
    loan = st.sidebar.selectbox("Personal Loan?", ['no', 'yes', 'unknown'])
    
    # Campaign Information
    st.sidebar.subheader("Campaign Details")
    contact = st.sidebar.selectbox("Contact Type", ['cellular', 'telephone'])
    month = st.sidebar.selectbox("Last Contact Month", [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ])
    day_of_week = st.sidebar.selectbox("Last Contact Day", 
        ['mon', 'tue', 'wed', 'thu', 'fri'])
    
    campaign = st.sidebar.number_input("Contacts in Current Campaign", 1, 50, 2)
    pdays = st.sidebar.number_input("Days Since Last Contact (999 if never)", 0, 999, 999)
    previous = st.sidebar.number_input("Previous Contacts", 0, 10, 0)
    poutcome = st.sidebar.selectbox("Previous Campaign Outcome", 
        ['nonexistent', 'failure', 'success'])
    
    # Economic Indicators
    st.sidebar.subheader("Economic Indicators")
    emp_var_rate = st.sidebar.number_input("Employment Variation Rate", -5.0, 5.0, 1.1, 0.1)
    cons_price_idx = st.sidebar.number_input("Consumer Price Index", 90.0, 100.0, 93.994, 0.001)
    cons_conf_idx = st.sidebar.number_input("Consumer Confidence Index", -60.0, 0.0, -36.4, 0.1)
    euribor3m = st.sidebar.number_input("Euribor 3 Month Rate", 0.0, 10.0, 4.857, 0.001)
    nr_employed = st.sidebar.number_input("Number of Employees", 4900.0, 5300.0, 5191.0, 0.1)
    
    # Create feature dictionary
    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([data])
    
    # Engineer features (MUST match the 3 features used in training: was_previously_contacted, campaign_successful, poutcome_success)
    input_df['was_previously_contacted'] = (input_df['pdays'] != 999).astype(int)
 
    input_df['campaign_successful'] = input_df['campaign'].apply(lambda x: 1 if x < 5 else 0) 
    
    input_df['poutcome_success'] = (input_df['poutcome'] == 'success').astype(int)
  
    return input_df

def main():
    # Header
    st.title("🏦 Bank Term Deposit Subscription Predictor")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Get user input
    input_df = get_user_input()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Client Profile Summary")
        
        # Displaying selected information
        st.markdown(f"""
        **Personal Information:**
        - Age: {input_df['age'].values[0]} years
        - Job: {input_df['job'].values[0]}
        - Education: {input_df['education'].values[0]}
        - Marital Status: {input_df['marital'].values[0]}
        
        **Financial Status:**
        - Housing Loan: {input_df['housing'].values[0]}
        - Personal Loan: {input_df['loan'].values[0]}
        - Credit Default: {input_df['default'].values[0]}
        
        **Campaign History:**
        - Contacts this campaign: {input_df['campaign'].values[0]}
        - Previous contacts: {input_df['previous'].values[0]}
        - Last outcome: {input_df['poutcome'].values[0]}
        """)
    



    with col2:
        st.subheader("🎯 Prediction")
        
        if st.button("🔮 Predict Subscription"):
            try:
                with st.spinner("Analyzing client profile..."):
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    probability = model.predict_proba(input_df)[0, 1]
                
                # Display results
                if prediction == 1:
                    st.success("### ✅ LIKELY TO SUBSCRIBE")
                    st.balloons()
                else:
                    st.error("### ❌ UNLIKELY TO SUBSCRIBE")
                
                # Probability gauge
                st.metric("Subscription Probability", f"{probability:.1%}")
                
                # Progress bar
                st.progress(probability)
                
                # Recommendation
                st.markdown("---")
                st.subheader("💡 Recommendation")
                
                if probability > 0.7:
                    st.success("""
                    **HIGH PRIORITY CLIENT**
                    - Immediate follow-up recommended
                    - Offer premium term deposit rates
                    - Assign to senior sales representative
                    """)
                elif probability > 0.4:
                    st.warning("""
                    **MEDIUM PRIORITY CLIENT**
                    - Standard marketing protocol
                    - Consider personalized offers
                    - Monitor engagement
                    """)
                else:
                    st.info("""
                    **LOW PRIORITY CLIENT**
                    - Reduce contact frequency
                    - Focus on relationship building
                    - Consider alternative products
                    """)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Please check that all required features are provided.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Bank Marketing Prediction System | ADA 442 Statistical Learning Project </p>
                <p> made by Ecem Nur Bilgi and İpek Özdemir  </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()

    #streamlit run app.py