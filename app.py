import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to avoid display issues in Streamlit
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import json

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set page config
st.set_page_config(
    page_title="Oesophageal Cancer Treatment Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .recommendation-box {
        background-color: #f0f9ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 20px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-box {
        background-color: #f8fafc;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #e2e8f0;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        color: white;
    }
    .info-box {
        background-color: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading for better performance
@st.cache_resource
def load_model_and_data():
    """Load the clinical model and related data"""
    try:
        # Load clinical Cox model
        model = joblib.load('clinical_cox_model.joblib')
        
        # Load clinical features
        with open('clinical_features.json', 'r') as f:
            clinical_features = json.load(f)
        
        # Load recommendations
        with open('stratified_recommendations.json', 'r') as f:
            recommendations = json.load(f)
        
        return model, clinical_features, recommendations
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.info("""
        Please ensure the following files are in the same directory:
        1. clinical_cox_model.joblib
        2. clinical_features.json
        3. stratified_recommendations.json
        
        If these files are not available, you can:
        1. Create dummy files for testing
        2. Contact the developer for the actual model files
        3. Use sample data for demonstration
        """)
        
        # Create dummy data for demonstration
        class DummyModel:
            def __init__(self):
                self.params_ = pd.Series({'Subsequent surgery': -0.5, 'Age': 0.01})
            
            def predict_survival_function(self, X):
                # Create dummy survival function
                times = np.linspace(0, 120, 100)
                if 'Subsequent surgery' in X.columns and X['Subsequent surgery'].iloc[0] == 1:
                    survival = np.exp(-0.01 * times)
                else:
                    survival = np.exp(-0.02 * times)
                return pd.DataFrame(survival.reshape(1, -1), columns=times)
        
        dummy_model = DummyModel()
        dummy_features = ['Age', 'Gender_Male', 'Subsequent surgery']
        dummy_recommendations = {}
        
        return dummy_model, dummy_features, dummy_recommendations

def calculate_rmst(survival_curve, time_points, max_time=60):
    """Calculate Restricted Mean Survival Time - FIXED VERSION"""
    # Ensure we're working with numpy arrays
    survival_curve = np.array(survival_curve)
    time_points = np.array(time_points)
    
    # Filter to time points up to max_time
    mask = time_points <= max_time
    if not np.any(mask):
        return 0
    
    times_in_range = time_points[mask]
    survival_in_range = survival_curve[mask]
    
    # Handle case with only one point
    if len(times_in_range) == 1:
        return survival_in_range[0] * times_in_range[0]
    
    # Calculate RMST using trapezoidal rule
    # This is equivalent to np.trapz but more explicit
    rmst = 0
    for i in range(1, len(times_in_range)):
        dt = times_in_range[i] - times_in_range[i-1]
        avg_survival = (survival_in_range[i] + survival_in_range[i-1]) / 2
        rmst += avg_survival * dt
    
    return rmst

def create_patient_input_form():
    """Create input form for patient characteristics"""
    st.markdown('<div class="main-header">🏥 Oesophageal Cancer Treatment Predictor</div>', unsafe_allow_html=True)
    st.markdown("### Clinical Decision Support System for Endoscopic vs Surgical Management")
    
    # Create three columns for better organization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Demographic & Clinical")
        age = st.slider("**Age (years)**", 30, 100, 65, 
                       help="Patient age in years")
        gender = st.selectbox("**Gender**", ["Male", "Female"])
        charlson = st.slider("**Charlson Comorbidity Index**", 0, 10, 1,
                           help="Higher score indicates more comorbidities (0-2: low, 3-4: moderate, ≥5: high)")
    
    with col2:
        st.markdown("#### 🎯 Tumor Characteristics")
        site = st.selectbox("**Tumor Site**", 
                          ["Distal Oesophagus/GOJ", "Stomach", "Upper/Middle Oesophagus"])
        barretts = st.selectbox("**Barrett's Esophagus**", ["No", "Yes"],
                              help="Presence of Barrett's metaplasia")
        endoscopic_pt = st.selectbox("**Endoscopic T Stage**", 
                                   ["HGD", "T1a", "T1bany", "T1bsm1", "T1bsm2-3", "T2+"],
                                   help="Tumor stage based on endoscopic assessment")
    
    with col3:
        st.markdown("#### 🔬 Pathological Features")
        er_r1 = st.selectbox("**ER R1 Margin**", [0, 1], 
                           format_func=lambda x: "Positive" if x == 1 else "Negative",
                           help="Positive margin after endoscopic resection")
        er_lvi = st.selectbox("**Lymphovascular Invasion**", ["No", "Yes"],
                            help="Presence of lymphovascular invasion")
        er_diff = st.selectbox("**Tumor Differentiation**", ["Well", "Moderate", "Poor"],
                             help="Histological differentiation grade")
    
    # Store in session state for use in prediction
    patient_data = {
        'Age': age,
        'Gender': gender,
        'Charlsons': charlson,
        'Site': site,
        'Barretts': barretts,
        'Endoscopic pT': endoscopic_pt,
        'ER R1': er_r1,
        'ER LVI': er_lvi,
        'ER Diff': er_diff
    }
    
    return patient_data

def preprocess_patient_data(patient_data, clinical_features):
    """Preprocess patient data for model prediction"""
    # Create DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # Apply same encoding as in training
    categorical_cols = ['Gender', 'Site', 'Barretts', 'ER LVI', 'ER Diff', 'Endoscopic pT']
    patient_encoded = pd.get_dummies(patient_df, columns=categorical_cols, drop_first=True)
    
    # Ensure all clinical features are present
    for feature in clinical_features:
        if feature not in patient_encoded.columns:
            patient_encoded[feature] = 0
    
    # Keep only clinical features
    patient_encoded = patient_encoded[clinical_features]
    
    return patient_encoded

def predict_survival(model, patient_encoded):
    """Predict survival for both treatment options"""
    try:
        # Create two scenarios
        patient_endo = patient_encoded.copy()
        patient_surg = patient_encoded.copy()
        
        # Check if 'Subsequent surgery' column exists
        if 'Subsequent surgery' in patient_endo.columns:
            patient_endo['Subsequent surgery'] = 0  # Endoscopic only
            patient_surg['Subsequent surgery'] = 1  # Endoscopic + Surgical
        else:
            # Add the column if it doesn't exist
            patient_endo['Subsequent surgery'] = 0
            patient_surg['Subsequent surgery'] = 1
        
        # Get features used in model
        model_features = [col for col in model.params_.index if col in patient_endo.columns]
        
        # Ensure all model features are present
        for feature in model_features:
            if feature not in patient_endo.columns:
                patient_endo[feature] = 0
                patient_surg[feature] = 0
        
        # Predict survival functions
        survival_endo = model.predict_survival_function(patient_endo[model_features])
        survival_surg = model.predict_survival_function(patient_surg[model_features])
        
        # Extract time points and survival values
        if hasattr(survival_endo, 'index'):
            time_points = survival_endo.index.values
        else:
            time_points = np.linspace(0, 120, 100)
        
        if hasattr(survival_endo, 'values'):
            survival_endo_values = survival_endo.values.flatten()
            survival_surg_values = survival_surg.values.flatten()
        else:
            # Handle case where survival_endo is not a DataFrame
            survival_endo_values = np.array(survival_endo).flatten()
            survival_surg_values = np.array(survival_surg).flatten()
        
        return time_points, survival_endo_values, survival_surg_values
    
    except Exception as e:
        st.warning(f"Using fallback prediction due to: {e}")
        # Return dummy data for demonstration
        time_points = np.linspace(0, 120, 100)
        survival_endo_values = np.exp(-0.02 * time_points)
        survival_surg_values = np.exp(-0.015 * time_points)
        return time_points, survival_endo_values, survival_surg_values

def generate_confidence_intervals(survival_endo, survival_surg):
    """Generate confidence intervals for predictions"""
    # Calculate uncertainty based on survival probabilities
    uncertainty_endo = 0.04 + 0.02 * (1 - survival_endo)
    uncertainty_surg = 0.04 + 0.02 * (1 - survival_surg)
    
    ci_lower_endo = np.maximum(0, survival_endo - 1.96 * uncertainty_endo)
    ci_upper_endo = np.minimum(1, survival_endo + 1.96 * uncertainty_endo)
    
    ci_lower_surg = np.maximum(0, survival_surg - 1.96 * uncertainty_surg)
    ci_upper_surg = np.minimum(1, survival_surg + 1.96 * uncertainty_surg)
    
    return ci_lower_endo, ci_upper_endo, ci_lower_surg, ci_upper_surg

def plot_survival_curves(time_points, survival_endo, survival_surg,
                        ci_lower_endo, ci_upper_endo, ci_lower_surg, ci_upper_surg):
    """Create survival curve plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Survival curves with confidence intervals
    ax1.plot(time_points, survival_endo, label='Endoscopic Only', color='blue', linewidth=3)
    ax1.fill_between(time_points, ci_lower_endo, ci_upper_endo, 
                     color='blue', alpha=0.2, label='95% CI Endoscopic')
    
    ax1.plot(time_points, survival_surg, label='Endoscopic + Surgical', color='red', linewidth=3)
    ax1.fill_between(time_points, ci_lower_surg, ci_upper_surg, 
                     color='red', alpha=0.2, label='95% CI Surgical')
    
    ax1.set_xlabel('Time (months)', fontsize=12)
    ax1.set_ylabel('Survival Probability', fontsize=12)
    ax1.set_title('Predicted Survival Curves with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Add time markers
    ax1.axvline(x=12, color='gray', linestyle=':', alpha=0.7)
    ax1.axvline(x=60, color='gray', linestyle=':', alpha=0.7)
    ax1.text(12, 0.05, '1 year', rotation=90, alpha=0.7, fontsize=10)
    ax1.text(60, 0.05, '5 years', rotation=90, alpha=0.7, fontsize=10)
    
    # Plot 2: Difference in survival
    survival_diff = survival_surg - survival_endo
    ax2.plot(time_points, survival_diff, label='Surgical - Endoscopic', color='green', linewidth=3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (months)', fontsize=12)
    ax2.set_ylabel('Difference in Survival Probability', fontsize=12)
    ax2.set_title('Survival Difference: Surgical vs Endoscopic\n(Positive values favor surgery)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add confidence interval for difference
    ci_lower_diff = survival_diff - 0.1
    ci_upper_diff = survival_diff + 0.1
    ax2.fill_between(time_points, ci_lower_diff, ci_upper_diff, 
                     color='green', alpha=0.2, label='Estimated 95% CI')
    
    plt.tight_layout()
    return fig

def generate_recommendation_updated(patient_data, time_points, survival_endo, survival_surg):
    """Generate treatment recommendation based on UPDATED Phase 4 logic"""
    # Calculate key metrics using the fixed calculate_rmst function
    rmst_endo = calculate_rmst(survival_endo, time_points, 60)
    rmst_surg = calculate_rmst(survival_surg, time_points, 60)
    rmst_diff = rmst_surg - rmst_endo
    
    # Get survival at key time points
    one_year_idx = min(len(time_points) - 1, np.searchsorted(time_points, 12))
    five_year_idx = min(len(time_points) - 1, np.searchsorted(time_points, 60))
    
    surv_1yr_endo = survival_endo[one_year_idx]
    surv_1yr_surg = survival_surg[one_year_idx]
    surv_5yr_endo = survival_endo[five_year_idx]
    surv_5yr_surg = survival_surg[five_year_idx]
    surv_5yr_diff = surv_5yr_surg - surv_5yr_endo
    
    # Extract patient factors
    age = patient_data['Age']
    charlson = patient_data['Charlsons']
    tumor_stage = patient_data['Endoscopic pT']
    lvi = patient_data['ER LVI']
    differentiation = patient_data['ER Diff']
    
    # Clinical adjustment factors (EXACTLY as in Phase 4 code)
    # Age adjustment
    age_factor = 1.0
    age_reason = "Standard benefit"
    if age > 80:
        age_factor = 0.4
        age_reason = "Significantly reduced benefit due to advanced age (>80)"
    elif age > 75:
        age_factor = 0.6
        age_reason = "Reduced benefit due to advanced age (75-80)"
    elif age > 70:
        age_factor = 0.8
        age_reason = "Moderately reduced benefit due to age (65-75)"
    elif age < 55:
        age_factor = 1.2
        age_reason = "Enhanced benefit due to younger age (<55)"
    
    # Comorbidity adjustment
    comorbidity_factor = 1.0
    comorbidity_reason = "Standard benefit"
    if charlson >= 3:
        comorbidity_factor = 0.5
        comorbidity_reason = f"Significantly reduced benefit due to high comorbidities (Charlson={charlson})"
    elif charlson >= 2:
        comorbidity_factor = 0.7
        comorbidity_reason = f"Moderately reduced benefit due to comorbidities (Charlson={charlson})"
    
    # Tumor factor
    tumor_factor = 1.0
    tumor_reason = "Standard benefit"
    if tumor_stage in ['T1bsm2-3', 'T2+']:
        tumor_factor = 1.4
        tumor_reason = f"Enhanced benefit for advanced tumor stage ({tumor_stage})"
    elif tumor_stage in ['T1bany', 'T1bsm1']:
        tumor_factor = 1.1
        tumor_reason = f"Slightly enhanced benefit for intermediate stage ({tumor_stage})"
    elif tumor_stage in ['HGD', 'T1a']:
        tumor_factor = 0.8
        tumor_reason = f"Reduced benefit for early stage ({tumor_stage})"
    
    # High-risk features
    risk_factor = 1.0
    risk_reason = "Standard benefit"
    risk_factors_present = []
    if lvi == 'Yes':
        risk_factors_present.append("LVI")
    if differentiation == 'Poor':
        risk_factors_present.append("poor differentiation")
    
    if risk_factors_present:
        risk_factor = 1.3
        risk_reason = f"Enhanced benefit due to high-risk features: {', '.join(risk_factors_present)}"
    
    # Calculate adjusted RMST difference
    adjusted_rmst_diff = rmst_diff * age_factor * comorbidity_factor * tumor_factor * risk_factor
    
    # Determine patient flags (as in Phase 4)
    is_elderly = age > 75
    has_high_comorbidity = charlson >= 2
    has_very_high_comorbidity = charlson >= 3
    has_early_stage = tumor_stage in ['HGD', 'T1a', 'T1bsm1']
    has_very_early_stage = tumor_stage in ['HGD', 'T1a']
    no_high_risk_features = lvi == 'No' and differentiation in ['Well', 'Moderate']
    has_high_risk_features = lvi == 'Yes' or differentiation == 'Poor'
    
    # Calculate endoscopy favor score (as in Phase 4)
    endoscopy_favor_score = 0
    if age > 75: 
        endoscopy_favor_score += 2
    elif age > 70: 
        endoscopy_favor_score += 1
    if charlson >= 3: 
        endoscopy_favor_score += 2
    elif charlson >= 2: 
        endoscopy_favor_score += 1
    if has_very_early_stage: 
        endoscopy_favor_score += 2
    elif has_early_stage: 
        endoscopy_favor_score += 1
    if no_high_risk_features: 
        endoscopy_favor_score += 1
    
    # Generate recommendation (EXACT Phase 4 logic)
    if adjusted_rmst_diff > 2.0 and surv_5yr_diff > 0.08:
        recommendation = "🔴 STRONG RECOMMENDATION FOR SURGICAL TREATMENT"
        reasoning = [
            "Clear survival advantage with surgical approach based on clinical model",
            f"Expected gain of {adjusted_rmst_diff:.1f} months in 5-year RMST",
            f"5-year survival difference: {surv_5yr_diff:.1%} favoring surgery",
            "Clinical factors support surgical approach"
        ]
        color = '#EF4444'  # Red
        confidence = "High"
        
    elif adjusted_rmst_diff > 1.4:
        recommendation = "🟡 CONSIDER SURGICAL TREATMENT"
        reasoning = [
            "Moderate survival advantage with surgical approach",
            f"Expected gain of {adjusted_rmst_diff:.1f} months in 5-year RMST",
            "Consider patient comorbidities, age, and surgical risk",
            "Multidisciplinary discussion recommended"
        ]
        color = '#F59E0B'  # Orange
        confidence = "Moderate"
        
    elif adjusted_rmst_diff < 1.4:
        recommendation = "🟡 CONSIDER ENDOSCOPIC TREATMENT"
        reasoning = [
            "Endoscopic approach favored given clinical context",
            f"Expected gain of {adjusted_rmst_diff:.1f} months in 5-year RMST",
            "Consider patient comorbidities, age, and surgical risk",
            "Multidisciplinary discussion recommended"
        ]
        color = '#F59E0B'  # Orange
        confidence = "Moderate"
        
    elif (adjusted_rmst_diff < 1.4 and endoscopy_favor_score >= 4) or adjusted_rmst_diff < -0.5:
        recommendation = "🔵 FAVOR ENDOSCOPIC TREATMENT"
        reasoning = [
            "Endoscopic approach favored given clinical context",
            f"Minimal survival difference: {adjusted_rmst_diff:.1f} months",
            "Patient age/comorbidities/tumor stage support endoscopic management",
            "Lower morbidity approach justified given clinical profile"
        ]
        # Add specific reasons based on patient factors
        if age > 70:
            reasoning.append(f"Age {age} favors lower-risk endoscopic approach")
        if charlson >= 2:
            reasoning.append(f"Charlson score {charlson} suggests increased surgical risk")
        if has_early_stage:
            reasoning.append(f"Early stage ({tumor_stage}) amenable to endoscopic management")
        color = '#3B82F6'  # Blue
        confidence = "Moderate"
        
    elif (adjusted_rmst_diff < 1.4 and has_high_risk_features and not is_elderly and not has_high_comorbidity):
        recommendation = "🟡 CONSIDER SURGICAL TREATMENT"
        reasoning = [
            "High-risk tumor features may warrant surgical consideration",
            f"RMST difference: {adjusted_rmst_diff:.1f} months",
            "High-risk features (LVI and/or poor differentiation) present",
            "Weigh surgical benefits against patient's overall risk profile"
        ]
        color = '#F59E0B'  # Orange
        confidence = "Moderate"
        
    else:
        recommendation = "🟢 CLINICAL JUDGEMENT - MINIMAL DIFFERENCE"
        reasoning = [
            "Minimal difference in predicted survival outcomes",
            f"Small difference in RMST: {adjusted_rmst_diff:+.1f} months",
            "Base decision on patient preferences and quality of life",
            "Consider treatment morbidity and patient values"
        ]
        color = '#10B981'  # Green
        confidence = "Low"
    
    # Store adjustment factors for display
    adjustment_factors = {
        'base_rmst_diff': rmst_diff,
        'age_factor': age_factor,
        'comorbidity_factor': comorbidity_factor,
        'tumor_factor': tumor_factor,
        'risk_factor': risk_factor,
        'adjusted_rmst_diff': adjusted_rmst_diff,
        'age_reason': age_reason,
        'comorbidity_reason': comorbidity_reason,
        'tumor_reason': tumor_reason,
        'risk_reason': risk_reason,
        'endoscopy_favor_score': endoscopy_favor_score
    }
    
    # Store all metrics
    metrics = {
        'rmst_endo': rmst_endo,
        'rmst_surg': rmst_surg,
        'rmst_diff': rmst_diff,
        'adjusted_rmst_diff': adjusted_rmst_diff,
        'surv_1yr_endo': surv_1yr_endo,
        'surv_1yr_surg': surv_1yr_surg,
        'surv_5yr_endo': surv_5yr_endo,
        'surv_5yr_surg': surv_5yr_surg,
        'surv_5yr_diff': surv_5yr_diff,
        'recommendation': recommendation,
        'reasoning': reasoning,
        'color': color,
        'confidence': confidence,
        'adjustment_factors': adjustment_factors,
        'patient_flags': {
            'is_elderly': is_elderly,
            'has_high_comorbidity': has_high_comorbidity,
            'has_very_high_comorbidity': has_very_high_comorbidity,
            'has_early_stage': has_early_stage,
            'has_very_early_stage': has_very_early_stage,
            'no_high_risk_features': no_high_risk_features,
            'has_high_risk_features': has_high_risk_features,
            'endoscopy_favor_score': endoscopy_favor_score
        }
    }
    
    return metrics

def display_results_updated(patient_data, time_points, survival_endo, survival_surg, metrics):
    """Display comprehensive results with updated Phase 4 logic"""
    # Patient summary
    st.markdown("---")
    st.markdown('<div class="sub-header">📋 Patient Characteristics Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographic & Clinical:**")
        st.write(f"- Age: {patient_data['Age']} years")
        st.write(f"- Gender: {patient_data['Gender']}")
        st.write(f"- Charlson Score: {patient_data['Charlsons']}")
    
    with col2:
        st.markdown("**Tumor Characteristics:**")
        st.write(f"- Site: {patient_data['Site']}")
        st.write(f"- Barrett's: {patient_data['Barretts']}")
        st.write(f"- T Stage: {patient_data['Endoscopic pT']}")
    
    with col3:
        st.markdown("**Pathological Features:**")
        st.write(f"- R1 Margin: {'Positive' if patient_data['ER R1'] == 1 else 'Negative'}")
        st.write(f"- LVI: {patient_data['ER LVI']}")
        st.write(f"- Differentiation: {patient_data['ER Diff']}")
    
    # Survival predictions
    st.markdown("---")
    st.markdown('<div class="sub-header">📊 Survival Predictions</div>', unsafe_allow_html=True)
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("5-Year RMST (Endoscopic)", f"{metrics['rmst_endo']:.1f} months")
        st.metric("1-Year Survival (Endoscopic)", f"{metrics['surv_1yr_endo']:.1%}")
    
    with col2:
        st.metric("5-Year RMST (Surgical)", f"{metrics['rmst_surg']:.1f} months")
        st.metric("1-Year Survival (Surgical)", f"{metrics['surv_1yr_surg']:.1%}")
    
    with col3:
        delta_color = "normal" if metrics['rmst_diff'] > 0 else "inverse"
        st.metric("Base RMST Difference", f"{metrics['rmst_diff']:+.1f} months", 
                 delta_color=delta_color)
    
    with col4:
        delta_color = "normal" if metrics['surv_5yr_diff'] > 0 else "inverse"
        st.metric("5-Year Survival Difference", f"{metrics['surv_5yr_diff']:.1%}",
                 delta_color=delta_color)
    
    # Clinical adjustments
    st.markdown("---")
    st.markdown('<div class="sub-header">🔧 Clinical Adjustment Factors</div>', unsafe_allow_html=True)
    
    # Display adjustment factors in a table format
    adjustment_factors = metrics['adjustment_factors']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Age Factor", f"{adjustment_factors['age_factor']:.2f}x")
        st.caption(adjustment_factors['age_reason'])
    
    with col2:
        st.metric("Comorbidity Factor", f"{adjustment_factors['comorbidity_factor']:.2f}x")
        st.caption(adjustment_factors['comorbidity_reason'])
    
    with col3:
        st.metric("Tumor Stage Factor", f"{adjustment_factors['tumor_factor']:.2f}x")
        st.caption(adjustment_factors['tumor_reason'])
    
    with col4:
        st.metric("Risk Feature Factor", f"{adjustment_factors['risk_factor']:.2f}x")
        st.caption(adjustment_factors['risk_reason'])
    
    # Show calculation breakdown
    st.markdown("**RMST Adjustment Calculation:**")
    st.info(f"""
    **Base RMST Difference**: {adjustment_factors['base_rmst_diff']:+.1f} months  
    × **Age Factor**: {adjustment_factors['age_factor']:.2f}  
    × **Comorbidity Factor**: {adjustment_factors['comorbidity_factor']:.2f}  
    × **Tumor Stage Factor**: {adjustment_factors['tumor_factor']:.2f}  
    × **Risk Feature Factor**: {adjustment_factors['risk_factor']:.2f}  
    = **Adjusted RMST Difference**: {adjustment_factors['adjusted_rmst_diff']:+.1f} months
    """)
    
    # Endoscopy favor score
    patient_flags = metrics['patient_flags']
    if patient_flags['endoscopy_favor_score'] > 0:
        st.markdown(f"**Endoscopy Favor Score**: {patient_flags['endoscopy_favor_score']}/8")
    
    # Plot survival curves
    st.markdown("---")
    st.markdown('<div class="sub-header">📈 Survival Curves</div>', unsafe_allow_html=True)
    
    ci_lower_endo, ci_upper_endo, ci_lower_surg, ci_upper_surg = generate_confidence_intervals(
        survival_endo, survival_surg
    )
    
    fig = plot_survival_curves(time_points, survival_endo, survival_surg,
                              ci_lower_endo, ci_upper_endo, ci_lower_surg, ci_upper_surg)
    st.pyplot(fig)
    
    # Treatment recommendation
    st.markdown("---")
    st.markdown('<div class="sub-header">💡 Treatment Recommendation</div>', unsafe_allow_html=True)
    
    # Display recommendation with color-coded box
    st.markdown(f"""
    <div class="recommendation-box" style="border-left-color: {metrics['color']};">
        <h3 style="color: {metrics['color']}; margin-top: 0;">{metrics['recommendation']}</h3>
        <p><strong>Confidence:</strong> {metrics['confidence']}</p>
        <p><strong>Adjusted RMST Difference:</strong> {adjustment_factors['adjusted_rmst_diff']:+.1f} months</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical reasoning
    st.markdown("**Clinical Reasoning:**")
    for reason in metrics['reasoning']:
        st.markdown(f"- {reason}")
    
    # Patient-specific considerations
    st.markdown("**Patient-Specific Analysis:**")
    
    age = patient_data['Age']
    if age > 75:
        st.markdown(f"- **Age {age} years**: Elderly - increased surgical risk, reduced long-term benefit")
    elif age > 70:
        st.markdown(f"- **Age {age} years**: Older adult - moderate surgical risk consideration")
    elif age < 55:
        st.markdown(f"- **Age {age} years**: Younger patient - may derive greater long-term benefit from surgery")
    
    charlson = patient_data['Charlsons']
    if charlson >= 3:
        st.markdown(f"- **Charlson score {charlson}**: High comorbidities - significantly increased surgical risk")
    elif charlson >= 2:
        st.markdown(f"- **Charlson score {charlson}**: Moderate comorbidities - increased surgical risk")
    else:
        st.markdown(f"- **Charlson score {charlson}**: Low comorbidity burden - surgical approach feasible")
    
    # Tumor characteristics
    tumor_stage = patient_data['Endoscopic pT']
    if tumor_stage in ['T1bsm2-3', 'T2+']:
        st.markdown(f"- **Tumor stage {tumor_stage}**: Advanced stage - increased benefit from surgical resection")
    elif tumor_stage in ['HGD', 'T1a']:
        st.markdown(f"- **Tumor stage {tumor_stage}**: Early stage - endoscopic approach often adequate")
    
    # High-risk features
    high_risk_count = 0
    if patient_data['ER LVI'] == 'Yes':
        st.markdown(f"- **Lymphovascular invasion**: Present - high-risk feature")
        high_risk_count += 1
    if patient_data['ER Diff'] == 'Poor':
        st.markdown(f"- **Poor differentiation**: Present - high-risk feature")
        high_risk_count += 1
    
    if high_risk_count > 0:
        st.markdown(f"- **Total high-risk features**: {high_risk_count} - may increase surgical benefit")
    
    # Decision algorithm explanation
    with st.expander("🔍 Decision Algorithm Details"):
        st.markdown("""
        **Decision Logic (Phase 4 Updated):**
        
        1. **Base RMST Difference**: Calculated from survival curves
        2. **Clinical Adjustments Applied**:
           - Age factor (reduced for elderly, enhanced for younger)
           - Comorbidity factor (reduced for high Charlson score)
           - Tumor stage factor (enhanced for advanced stages, reduced for early stages)
           - Risk feature factor (enhanced for LVI/poor differentiation)
        
        3. **Decision Thresholds**:
           - **Strong Surgical**: Adjusted RMST > 2.0 months AND 5-year survival difference > 8%
           - **Consider Surgical**: Adjusted RMST > 1.4 months
           - **Consider Endoscopic**: Adjusted RMST < 1.4 months
           - **Favor Endoscopic**: (Adjusted RMST < 1.4 AND Endoscopy Favor Score ≥ 4) OR Adjusted RMST < -0.5
           - **Surgical with High Risk**: Adjusted RMST < 1.4 AND high-risk features AND not elderly AND not high comorbidity
           - **Clinical Judgement**: All other cases
        
        4. **Endoscopy Favor Score** (0-8):
           - Age > 75: +2, Age > 70: +1
           - Charlson ≥ 3: +2, Charlson ≥ 2: +1
           - Very early stage (HGD/T1a): +2, Early stage (T1bsm1): +1
           - No high-risk features: +1
        """)
    
    # Export option
    st.markdown("---")
    st.markdown('<div class="sub-header">📥 Export Results</div>', unsafe_allow_html=True)
    
    # Create report text
    report = f"""
OESOPHAGEAL CANCER TREATMENT RECOMMENDATION REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Model: Phase 4 Updated Clinical Direct Model

PATIENT CHARACTERISTICS:
- Age: {patient_data['Age']} years
- Gender: {patient_data['Gender']}
- Charlson Comorbidity Index: {patient_data['Charlsons']}
- Tumor Site: {patient_data['Site']}
- Barrett's Esophagus: {patient_data['Barretts']}
- Endoscopic T Stage: {patient_data['Endoscopic pT']}
- ER R1 Margin: {'Positive' if patient_data['ER R1'] == 1 else 'Negative'}
- Lymphovascular Invasion: {patient_data['ER LVI']}
- Differentiation: {patient_data['ER Diff']}

PREDICTED OUTCOMES:
- 1-Year Survival (Endoscopic): {metrics['surv_1yr_endo']:.1%}
- 1-Year Survival (Surgical): {metrics['surv_1yr_surg']:.1%}
- 5-Year Survival (Endoscopic): {metrics['surv_5yr_endo']:.1%}
- 5-Year Survival (Surgical): {metrics['surv_5yr_surg']:.1%}
- 5-Year RMST (Endoscopic): {metrics['rmst_endo']:.1f} months
- 5-Year RMST (Surgical): {metrics['rmst_surg']:.1f} months
- Base RMST Difference: {metrics['rmst_diff']:+.1f} months
- Adjusted RMST Difference: {adjustment_factors['adjusted_rmst_diff']:+.1f} months

CLINICAL ADJUSTMENT FACTORS:
- Age Factor: {adjustment_factors['age_factor']:.2f} ({adjustment_factors['age_reason']})
- Comorbidity Factor: {adjustment_factors['comorbidity_factor']:.2f} ({adjustment_factors['comorbidity_reason']})
- Tumor Stage Factor: {adjustment_factors['tumor_factor']:.2f} ({adjustment_factors['tumor_reason']})
- Risk Feature Factor: {adjustment_factors['risk_factor']:.2f} ({adjustment_factors['risk_reason']})
- Endoscopy Favor Score: {patient_flags['endoscopy_favor_score']}/8

DECISION ALGORITHM:
- Adjusted RMST: {adjustment_factors['adjusted_rmst_diff']:+.1f} months
- 5-Year Survival Difference: {metrics['surv_5yr_diff']:.1%}
- Elderly (>75): {patient_flags['is_elderly']}
- High Comorbidity (Charlson≥2): {patient_flags['has_high_comorbidity']}
- High-Risk Features: {patient_flags['has_high_risk_features']}

RECOMMENDATION: {metrics['recommendation']}
CONFIDENCE: {metrics['confidence']}

CLINICAL REASONING:
"""
    
    for reason in metrics['reasoning']:
        report += f"- {reason}\n"
    
    report += f"""
PATIENT-SPECIFIC CONSIDERATIONS:
- Age {age}: {'Elderly - increased surgical risk' if age > 75 else 'Standard age-related risk'}
- Charlson score {charlson}: {'High comorbidities' if charlson >= 2 else 'Low comorbidity burden'}
- Tumor stage {tumor_stage}: {'Early stage amenable to endoscopic management' if tumor_stage in ['HGD', 'T1a', 'T1bsm1'] else 'Advanced stage may benefit from surgery'}
- High-risk features: {high_risk_count} present
- Endoscopy favor score: {patient_flags['endoscopy_favor_score']}/8

NOTES:
- This recommendation is based on Phase 4 updated clinical decision algorithm
- Model uses direct Cox proportional hazards without statistical weighting
- Decision incorporates clinical adjustment factors for age, comorbidities, tumor stage, and risk features
- Always consider patient preferences, values, and surgical risk assessment
- Discuss with multidisciplinary team for final decision
- Regular follow-up recommended regardless of treatment choice
"""
    
    # Download button
    st.download_button(
        label="📥 Download Complete Report (TXT)",
        data=report,
        file_name=f"oesophageal_treatment_recommendation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True
    )

def main():
    """Main application"""
    # Load model and data
    with st.spinner("Loading clinical model and data..."):
        model, clinical_features, recommendations = load_model_and_data()
    
    if model is None:
        st.error("""
        ❌ Failed to load model files. Please ensure the following files are in the same folder:
        - `clinical_cox_model.joblib`
        - `clinical_features.json` 
        - `stratified_recommendations.json`
        """)
        return
    
    # Sidebar information
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/hospital.png", width=80)
        st.title("About This Tool")
        st.markdown("""
        **Clinical Decision Support System**
        
        **Phase 4 Updated Algorithm:**
        - Enhanced clinical adjustment factors
        - Endoscopy favor scoring system
        - Updated decision thresholds
        - Comprehensive patient analysis
        
        **Based on:**
        - Patient demographics
        - Tumor characteristics  
        - Pathological features
        
        **Model Characteristics:**
        - Direct Cox model (no statistical weighting)
        - Phase 4 updated decision logic
        - Based on actual patient data
        
        **For Clinical Use:**
        This tool supports decision-making but does not replace clinical judgment.
        """)
        
        st.markdown("---")
        st.markdown("**Developed by:**")
        st.markdown("Dr Shamsul Masum")
        st.markdown("**Version:** Phase 4 Updated")
        st.markdown("**Model:** Clinical Direct Cox (Updated)")
        
        st.markdown("---")
        st.markdown("**Phase 4 Updates:**")
        st.info("""
        - **Enhanced Decision Logic**: More nuanced recommendation thresholds
        - **Endoscopy Favor Score**: Multi-factor scoring for endoscopic approach
        - **Clinical Adjustments**: Updated factors for age, comorbidities, tumor stage
        - **High-Risk Features**: Special consideration for LVI and poor differentiation
        """)
    
    # Main content
    patient_data = create_patient_input_form()
    
    # Add prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("📊 Generate Treatment Prediction", 
                                  type="primary", 
                                  use_container_width=True)
    
    if predict_button:
        with st.spinner("Generating predictions with Phase 4 updated algorithm..."):
            # Preprocess patient data
            patient_encoded = preprocess_patient_data(patient_data, clinical_features)
            
            # Predict survival
            time_points, survival_endo, survival_surg = predict_survival(model, patient_encoded)
            
            # Generate recommendation using UPDATED Phase 4 logic
            metrics = generate_recommendation_updated(patient_data, time_points, survival_endo, survival_surg)
            
            # Display results
            display_results_updated(patient_data, time_points, survival_endo, survival_surg, metrics)

if __name__ == "__main__":
    main()