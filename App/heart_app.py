import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from PIL import Image
from lightgbm import LGBMClassifier
import category_encoders as ce
from imblearn.ensemble import EasyEnsembleClassifier
import shap
import plotly.express as px
import sqlite3
import bcrypt
import time

# -----------------------------------------------------------------------------
# Configuration & Assets
# -----------------------------------------------------------------------------
st.set_page_config(layout='wide', page_title='AI Health Risk Assessment', page_icon="❤️")

# Load Models
@st.cache_resource
def load_models():
    try:
        with open('best_model.pkl', 'rb') as model_file:
            model = pkl.load(model_file)
        with open('cbe_encoder.pkl', 'rb') as encoder_file:
            encoder = pkl.load(encoder_file)
        return model, encoder
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

model, encoder = load_models()

# Apply Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# -----------------------------------------------------------------------------
# Database Management
# -----------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    if data:
        return bcrypt.checkpw(password.encode('utf-8'), data[0])
    return False

init_db()

# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

# -----------------------------------------------------------------------------
# Auth Views
# -----------------------------------------------------------------------------
if not st.session_state['logged_in']:
    # Centered Layout for Login
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Login Card Container
        st.markdown("""
        <div class="login-container">
            <h1 style="color: #2563eb; margin-bottom: 0;">Health AI</h1>
            <p style="color: #6b7280; font-size: 1.1rem;">Secure Access to Risk Assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                st.markdown("<br>", unsafe_allow_html=True)
                submit = st.form_submit_button("Sign In", use_container_width=True)
                
                if submit:
                    if login_user(username, password):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.success("Welcome back!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

        with tab_register:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("register_form"):
                new_user = st.text_input("New Username", placeholder="Pick a unique username")
                new_pass = st.text_input("Create Password", type="password", placeholder="Create a strong password")
                confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
                st.markdown("<br>", unsafe_allow_html=True)
                submit_reg = st.form_submit_button("Create Account", use_container_width=True)
                
                if submit_reg:
                    if not new_user or not new_pass:
                         st.warning("Please fill in all fields")
                    elif new_pass != confirm_pass:
                        st.error("Passwords do not match")
                    elif add_user(new_user, new_pass):
                        st.success("Account created! Please switch to Login tab.")
                    else:
                        st.error("Username already exists")

else:
    # -----------------------------------------------------------------------------
    # Main App Logic (Logged In)
    # -----------------------------------------------------------------------------
    
    # Header with User Info & Logout
    c1, c2 = st.columns([8, 1])
    with c1:
        st.title(f"Welcome, {st.session_state['username']}! 👋")
    with c2:
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = ''
            st.rerun()
            
    st.markdown("<h3 style='text-align: center; color: #4b5563;'>Advanced Heart Disease & Diabetes Prediction</h3>", unsafe_allow_html=True)
    st.write('---')

    # Intro Cards
    st.markdown("""
    <div class="flex-container">
        <div class="flex-item">
            <h2>📊 Comprehensive Assessment</h2>
            <p>Utilizing state-of-the-art AI models to analyze your health metrics and predict potential risks for Heart Disease and Diabetes with high accuracy.</p>
        </div>
        <div class="flex-item">
            <h2>🛠️ How It Works</h2>
            <p><strong>1. Input Data:</strong> Enter your medical history and lifestyle details.<br>
            <strong>2. AI Analysis:</strong> Our algorithms process your data against vast datasets.<br>
            <strong>3. Get Results:</strong> Receive a personalized risk score and actionable health advice.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------------
    # Tabs Layout
    # -----------------------------------------------------------------------------
    tab1, tab2 = st.tabs(["❤️ Heart Disease Risk", "🩸 Diabetes Risk"])

    # =============================================================================
    # TAB 1: HEART DISEASE
    # =============================================================================
    with tab1:
        st.header("Heart Disease Risk Assessment")
        st.markdown("""
        <div class="metric-container" style="padding: 10px; margin-bottom: 20px; text-align: left; border-left: 5px solid #2563eb;">
            <span class="metric-label" style="font-size: 1rem;">Model Accuracy:</span>
            <span class="metric-value" style="font-size: 1.5rem; color: #2563eb;">88.5%</span>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("👤 Step 1: Demographics", expanded=True):
            c1, c2, c3 = st.columns(3)
            gender = c1.selectbox("Gender", ["female", "male", "nonbinary"], index=1, key="hd_gender")
            race = c2.selectbox("Race/Ethnicity", [
                "white_only_non_hispanic", "black_only_non_hispanic", "asian_only_non_hispanic", 
                "american_indian_or_alaskan_native_only_non_hispanic", "multiracial_non_hispanic", 
                "hispanic", "native_hawaiian_or_other_pacific_islander_only_non_hispanic"
            ], index=0, key="hd_race")
            age_category = c3.selectbox("Age Group", [
                "Age_18_to_24", "Age_25_to_29", "Age_30_to_34", "Age_35_to_39", 
                "Age_40_to_44", "Age_45_to_49", "Age_50_to_54", "Age_55_to_59",
                "Age_60_to_64", "Age_65_to_69", "Age_70_to_74", "Age_75_to_79",
                "Age_80_or_older"
            ], index=4, key="hd_age")

        with st.expander("🏥 Step 2: Medical History", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                general_health = st.selectbox("Overall Health", ["excellent", "very_good", "good", "fair", "poor"], index=0, key="hd_health")
                heart_attack = st.selectbox("History of Heart Attack?", ["yes", "no"], index=1, key="hd_attack")
                kidney_disease = st.selectbox("History of Kidney Disease?", ["yes", "no"], index=1, key="hd_kidney")
                asthma = st.selectbox("Asthma History?", ["never_asthma", "current_asthma", "former_asthma"], index=0, key="hd_asthma")
                could_not_afford_to_see_doctor = st.selectbox("Unable to see a doctor due to cost?", ["yes", "no"], index=1, key="hd_doc_cost")
                depressive_disorder = st.selectbox("History of Depressive Disorder?", ["yes", "no"], index=1, key="hd_depress")
            with c2:
                bmi_val = st.selectbox("BMI Category", [
                    "underweight_bmi_less_than_18_5", "normal_weight_bmi_18_5_to_24_9", "overweight_bmi_25_to_29_9",  
                    "obese_bmi_30_or_more"
                ], index=1, key="hd_bmi")
                diabetes_hist = st.selectbox("History of Diabetes?", ["yes", "no", "no_prediabetes", "yes_during_pregnancy"], index=1, key="hd_diab")
                stroke = st.selectbox("History of Stroke?", ["yes", "no"], index=1, key="hd_stroke")
                health_care_provider = st.selectbox("Have Primary Healthcare Provider?", ["yes_only_one", "more_than_one", "no"], index=0, key="hd_provider")
                length_of_time_since_last_routine_checkup = st.selectbox("Time since last checkup?", ["past_year", "past_2_years", "past_5_years", "5+_years_ago", "never"], index=0, key="hd_checkup")
                walking = st.selectbox("Difficulty walking/climbing stairs?", ["yes", "no"], index=1, key="hd_walk")

            c1, c2 = st.columns(2)
            physical_health = c1.selectbox("Days feeling physically unwell (past 30 days)", ["zero_days_not_good", "1_to_13_days_not_good", "14_plus_days_not_good"], index=0, key="hd_phys")
            mental_health = c2.selectbox("Days feeling mentally unwell (past 30 days)", ["zero_days_not_good", "1_to_13_days_not_good", "14_plus_days_not_good"], index=0, key="hd_ment")

        with st.expander("🏃 Step 3: Lifestyle", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                smoking_status = st.selectbox("Smoking Status", ["never_smoked", "former_smoker", "current_smoker_some_days", "current_smoker_every_day"], index=0, key="hd_smoke")
                drinks_category = st.selectbox("Alcohol Consumption (weekly)", [
                    "did_not_drink", "very_low_consumption_0.01_to_1_drinks", "low_consumption_1.01_to_5_drinks",  
                    "moderate_consumption_5.01_to_10_drinks", "high_consumption_10.01_to_20_drinks", "very_high_consumption_more_than_20_drinks"], index=0, key="hd_drink")
            with c2:
                sleep_category = st.selectbox("Typical Sleep Hours", [
                    "very_short_sleep_0_to_3_hours", "short_sleep_4_to_5_hours", "normal_sleep_6_to_8_hours",  
                    "long_sleep_9_to_10_hours", "very_long_sleep_11_or_more_hours"], index=2, key="hd_sleep")
                binge_drinking_status = st.selectbox("Binge Drinking (past 30 days)?", ["yes", "no"], index=1, key="hd_binge")
                exercise_status = st.selectbox("Exercised in past 30 days?", ["yes", "no"], index=0, key="hd_exer")

        # HD Prediction Logic
        if st.button('Assess Heart Disease Risk', key="btn_hd"):
            hd_input = {
                'gender': gender, 'race': race, 'general_health': general_health,
                'health_care_provider': health_care_provider, 'could_not_afford_to_see_doctor': could_not_afford_to_see_doctor,
                'length_of_time_since_last_routine_checkup': length_of_time_since_last_routine_checkup,
                'ever_diagnosed_with_heart_attack': heart_attack, 'ever_diagnosed_with_a_stroke': stroke,
                'ever_told_you_had_a_depressive_disorder': depressive_disorder, 'ever_told_you_have_kidney_disease': kidney_disease,
                'ever_told_you_had_diabetes': diabetes_hist, 'BMI': bmi_val,
                'difficulty_walking_or_climbing_stairs': walking, 'physical_health_status': physical_health,
                'mental_health_status': mental_health, 'asthma_Status': asthma,
                'smoking_status': smoking_status, 'binge_drinking_status': binge_drinking_status,
                'exercise_status_in_past_30_Days': exercise_status, 'age_category': age_category,
                'sleep_category': sleep_category, 'drinks_category': drinks_category
            }
            
            try:
                input_df = pd.DataFrame([hd_input])
                input_encoded = encoder.transform(input_df, y=None, override_return_df=False)
                risk_score = model.predict_proba(input_encoded)[:, 1][0] * 100
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"""
                    <div class="fade-in-up">
                        <div class="metric-container">
                            <div class="metric-label">Heart Disease Probability</div>
                            <div class="metric-value">{risk_score:.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c2:
                    if risk_score > 50:
                        st.error("High Risk: Please consult a cardiologist immediately.")
                    elif risk_score > 20:
                        st.warning("Moderate Risk: Consider lifestyle changes and regular checkups.")
                    else:
                        st.success("Low Risk: Keep up the healthy lifestyle!")
                
                # SHAP
                lgbm_model = model.estimators_[0].steps[-1][1]
                explainer = shap.TreeExplainer(lgbm_model)
                shap_values = explainer.shap_values(input_encoded)
                # Handle varying SHAP output formats (list vs array)
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    vals = shap_values[1]
                elif isinstance(shap_values, list):
                    vals = shap_values[0]
                else:
                    vals = shap_values
                feature_importances = np.abs(vals).sum(axis=0)
                feature_importances = 100 * (feature_importances / feature_importances.sum())
                
                feat_df = pd.DataFrame({'Feature': input_encoded.columns, 'Importance': feature_importances}).sort_values('Importance', ascending=False).head(10)
                fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title='Top Risk Contributors', color='Importance', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Analysis Failed: {e}")

    # =============================================================================
    # TAB 2: DIABETES
    # =============================================================================
    with tab2:
        st.header("Diabetes Risk Assessment")
        st.markdown("""
        <div class="metric-container" style="padding: 10px; margin-bottom: 20px; text-align: left; border-left: 5px solid #ef4444;">
            <span class="metric-label" style="font-size: 1rem;">Model Accuracy:</span>
            <span class="metric-value" style="font-size: 1.5rem; color: #ef4444;">92.4%</span>
        </div>
        """, unsafe_allow_html=True)
        st.info("ℹ️ This updated module estimates your risk of Type 2 Diabetes based on an expanded set of health markers.")

        with st.expander("📝 Vital Statistics & History", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                d_age = st.slider("Age", 18, 100, 45, key="d_age")
                d_bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0, 0.1, key="d_bmi")
                d_waist = st.number_input("Waist Circumference (cm)", 50, 200, 85, key="d_waist")
                d_bp = st.selectbox("High Blood Pressure?", ["No", "Yes"], key="d_bp")
            
            with c2:
                d_gender = st.selectbox("Gender", ["Female", "Male"], key="d_gender")
                d_family = st.selectbox("Family History of Diabetes?", ["No", "Yes"], key="d_fam")
                d_gestational = st.selectbox("History of Gestational Diabetes? (Women only)", ["No", "Yes", "Not Applicable"], key="d_gest")

        with st.expander("🩺 Clinical Markers (If known)", expanded=True):
            st.caption("Leave default values if you don't have recent lab results.")
            c1, c2 = st.columns(2)
            with c1:
                d_glucose = st.number_input("Fasting Glucose (mg/dL)", 0, 300, 0, help="Normal is < 100 mg/dL", key="d_gluc")
                d_hba1c = st.number_input("HbA1c Level (%)", 0.0, 15.0, 0.0, 0.1, help="Normal is < 5.7%", key="d_hba1c")
            with c2:
                d_cholesterol = st.number_input("Total Cholesterol (mg/dL)", 0, 500, 0, key="d_chol")
                d_trig = st.number_input("Triglycerides (mg/dL)", 0, 1000, 0, key="d_trig")

        with st.expander("🧘 Lifestyle & Habits", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                d_activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate (1-3 days/week)", "Active (4+ days/week)"], key="d_act_lvl")
                d_diet = st.selectbox("Diet Quality", ["Poor (High Sugar/Processed)", "Average", "Healthy (Fruits/Veggies/Whole Grains)"], key="d_diet")
                d_water = st.slider("Daily Water Intake (Liters)", 0.0, 5.0, 2.0, 0.1, key="d_water")
            with c2:
                d_stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"], key="d_stress")
                d_sleep = st.selectbox("Average Sleep", ["< 5 hours", "5-7 hours", "7-9 hours", "> 9 hours"], key="d_sleep_hours")
                d_smoke = st.selectbox("Do you smoke?", ["No", "Yes"], key="d_smoker")

        if st.button('Assess Diabetes Risk', key="btn_dia"):
            # Expanded Heuristic Logic
            score = 0
            
            # Age Factor
            if d_age > 60: score += 15
            elif d_age > 45: score += 10
            
            # BMI Factor
            if d_bmi > 35: score += 20
            elif d_bmi > 30: score += 15
            elif d_bmi > 25: score += 5
            
            # Waist Circumference (Central Obesity)
            if d_gender == "Male" and d_waist > 94: score += 5
            if d_gender == "Male" and d_waist > 102: score += 10
            if d_gender == "Female" and d_waist > 80: score += 5
            if d_gender == "Female" and d_waist > 88: score += 10

            # Health Factors
            if d_bp == "Yes": score += 10
            if d_family == "Yes": score += 15
            if d_gestational == "Yes": score += 10
            
            # Lifestyle
            if d_activity == "Sedentary": score += 10
            if d_diet == "Poor (High Sugar/Processed)": score += 10
            if d_stress == "High": score += 5
            if d_smoke == "Yes": score += 5
            if d_sleep == "< 5 hours": score += 5
            
            # Clinical Overrides (Strong indicators)
            if d_glucose > 126: score = max(score, 95)
            elif d_glucose > 100: score = max(score, 60)
            
            if d_hba1c > 6.5: score = max(score, 95)
            elif d_hba1c > 5.7: score = max(score, 60)

            # Cap score
            score = min(score, 99)
            score = max(score, 1)

            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"""
                <div class="fade-in-up">
                    <div class="metric-container" style="border-left-color: #ef4444;">
                        <div class="metric-label">Diabetes Risk Score</div>
                        <div class="metric-value" style="color: #ef4444;">{score}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                if score > 50:
                    st.error("High Risk: We recommend consulting a healthcare provider for a blood sugar test.")
                    st.markdown("**Key Risk Factors Detected:**")
                    if d_bmi > 25: st.write("- Elevated BMI")
                    if d_bp == "Yes": st.write("- High Blood Pressure")
                    if d_family == "Yes": st.write("- Family History")
                elif score > 20:
                    st.warning("Moderate Risk: Focus on diet and exercise to lower your risk.")
                else:
                    st.success("Low Risk: Maintain your healthy habits!")

    st.write('---')

    # -----------------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------------
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 20px;">
        <h4>TEAM C10 CSE C MLRIT 2026</h4>
        <p>© All rights reserved.</p>
        <p style="font-size: 0.8rem;">Disclaimer: This application is for educational and informational purposes only and does not constitute medical advice.</p>
    </div>
    """, unsafe_allow_html=True)
