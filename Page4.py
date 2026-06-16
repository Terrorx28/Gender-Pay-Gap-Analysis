import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.data_loader import load_data

# Calculate Job Level Gender Gaps dynamically using OLS regression
@st.cache_data
def calculate_job_level_gaps(df):
    gaps = {}
    for lvl in ['Junior', 'Mid', 'Senior', 'Manager', 'Director']:
        sub = df[df['job_level'] == lvl]
        if len(sub) > 10 and sub['gender'].nunique() >= 2:
            # Fit simple OLS for experience and gender within this job level
            X = pd.get_dummies(sub[['gender', 'years_of_experience']], drop_first=True)
            y = sub['monthly_salary_inr']
            gender_cols = [c for c in X.columns if 'gender_Male' in c]
            if gender_cols:
                lr = LinearRegression().fit(X, y)
                gaps[lvl] = lr.coef_[X.columns.get_loc(gender_cols[0])]
            else:
                gaps[lvl] = 0.0
        else:
            # Fallback values from regression averages
            gaps[lvl] = 2000.0 if lvl == 'Junior' else 5800.0 if lvl == 'Mid' else 8500.0 if lvl == 'Senior' else 13600.0 if lvl == 'Manager' else 20000.0
    return gaps

# Train ML Models - (1) Bias Model on all data, (2) Fair Model on Males only
@st.cache_resource
def train_page4_models(df):
    features_with_gender = [
        'gender', 'years_of_experience', 'job_level', 'industry', 'company_size',
        'negotiated_salary', 'employment_type', 'career_gap_months', 'num_promotions',
        'primary_caregiver', 'relocated_for_job', 'works_overtime', 'education_level',
        'city_tier', 'performance_rating', 'age', 'domain'
    ]
    
    features_no_gender = [f for f in features_with_gender if f != 'gender']
    
    categorical_cols = ['job_level', 'industry', 'company_size', 'employment_type', 
                        'education_level', 'city_tier', 'domain']
    
    # Preprocessors
    preprocessor_with_gender = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['gender'] + categorical_cols)
        ],
        remainder='passthrough'
    )
    
    preprocessor_no_gender = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Model Pipelines (Linear Regression used for high simulator responsiveness)
    pipeline_with_gender = Pipeline(steps=[
        ('preprocessor', preprocessor_with_gender),
        ('regressor', LinearRegression())
    ])
    
    pipeline_no_gender = Pipeline(steps=[
        ('preprocessor', preprocessor_no_gender),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Train full bias model on the whole dataset
    X_with = df[features_with_gender]
    y_with = df['monthly_salary_inr']
    pipeline_with_gender.fit(X_with, y_with)
    
    # Train counterfactual fair model on Males only (unbiased benchmark)
    df_males = df[df['gender'] == 'Male']
    X_no = df_males[features_no_gender]
    y_no = df_males['monthly_salary_inr']
    pipeline_no_gender.fit(X_no, y_no)
    
    return pipeline_with_gender, pipeline_no_gender, features_with_gender, features_no_gender

def get_feature_importances(pipeline, features_no_gender):
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']
    
    cat_encoder = preprocessor.named_transformers_['cat']
    categorical_cols = ['job_level', 'industry', 'company_size', 'employment_type', 
                        'education_level', 'city_tier', 'domain']
    
    encoded_cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    numeric_cols = [f for f in features_no_gender if f not in categorical_cols]
    
    all_feature_names = encoded_cat_names + numeric_cols
    importances = regressor.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    })
    
    grouped_importance = []
    for col in ['years_of_experience', 'job_level', 'industry', 'company_size', 'education_level', 'domain', 'career_gap_months', 'num_promotions', 'performance_rating', 'works_overtime', 'negotiated_salary']:
        imp_sum = importance_df[importance_df['Feature'].str.startswith(col)]['Importance'].sum()
        grouped_importance.append({'Feature': col.replace('_', ' ').title(), 'Importance': imp_sum})
        
    return pd.DataFrame(grouped_importance).sort_values(by='Importance', ascending=True)

def show():
    # Page Header
    st.markdown('<h1 class="main-title">🤖 PayEquity AI & Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-Powered Salary Predictor, Bias Auditor, and Workplace Policy Simulator Sandbox</p>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.error("Data could not be loaded. Please reload the main page.")
        return
        
    df = st.session_state.df
    
    # Train/Fetch Models and dynamic gaps
    with st.spinner("Training PayEquity AI Random Forest models..."):
        model_bias, model_fair, features_with, features_no = train_page4_models(df)
        job_gaps = calculate_job_level_gaps(df)
        
    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Profile Bias Simulator", 
        "🔍 Personal Fair Pay Audit", 
        "⚖️ EquiSim Policy Sandbox", 
        "📋 Corporate Strategy Generator"
    ])
    
    # ── TAB 1: PROFILE BIAS SIMULATOR ──────────────────────────────────────────
    with tab1:
        st.subheader("Gender Bias Profile Simulator")
        st.write(
            "Input credentials for a hypothetical employee. The AI calculates the market salary baseline, "
            "compares predictions for Male vs. Female, and reports the unexplained **Gender Discount**."
        )
        
        left_input, right_output = st.columns([1, 1])
        
        with left_input:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("#### 👤 Credentials")
            
            sim_exp = st.slider("Years of Experience", min_value=0.0, max_value=25.0, value=6.0, step=0.5, key="p4_exp")
            sim_age = st.slider("Age (Years)", min_value=20, max_value=60, value=28, key="p4_age")
            sim_level = st.selectbox("Job Level", options=['Junior', 'Mid', 'Senior', 'Manager', 'Director'], index=1, key="p4_lvl")
            sim_industry = st.selectbox("Industry", options=sorted(df['industry'].unique()), index=8, key="p4_ind")
            sim_domain = st.selectbox("Job Domain", options=sorted(df['domain'].unique()), index=2, key="p4_dom")
            sim_education = st.selectbox("Education Level", options=sorted(df['education_level'].unique()), index=0, key="p4_edu")
            sim_company = st.selectbox("Company Size", options=sorted(df['company_size'].unique()), key="p4_size")
            sim_city = st.selectbox("City Tier", options=sorted(df['city_tier'].unique()), key="p4_city")
            sim_rating = st.slider("Performance Rating (1-5)", min_value=1, max_value=5, value=3, key="p4_rate")
            
            st.write("#### 💼 Working Characteristics")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                sim_neg = st.checkbox("Negotiated starting salary", value=True, key="p4_neg")
                sim_ot = st.checkbox("Works overtime", value=False, key="p4_ot")
            with col_c2:
                sim_care = st.checkbox("Primary Caregiver", value=False, key="p4_care")
                sim_reloc = st.checkbox("Relocated for job", value=False, key="p4_reloc")
                
            sim_gap = st.number_input("Career break (months)", min_value=0, max_value=48, value=0, key="p4_gap")
            sim_promo = st.number_input("Number of promotions", min_value=0, max_value=6, value=1, key="p4_promo")
            
            neg_val = 1 if sim_neg else 0
            ot_val = 1 if sim_ot else 0
            care_val = 1 if sim_care else 0
            reloc_val = 1 if sim_reloc else 0
            st.markdown('</div>', unsafe_allow_html=True)
            
        with right_output:
            # Construct a profile to predict unbiased benchmark salary (Male schedule)
            profile_base = pd.DataFrame([{
                'years_of_experience': sim_exp, 'job_level': sim_level, 'industry': sim_industry, 
                'company_size': sim_company, 'negotiated_salary': neg_val, 'employment_type': 'Full-Time', 
                'career_gap_months': sim_gap, 'num_promotions': sim_promo, 'primary_caregiver': care_val, 
                'relocated_for_job': reloc_val, 'works_overtime': ot_val, 'education_level': sim_education, 
                'city_tier': sim_city, 'performance_rating': sim_rating, 'age': sim_age, 'domain': sim_domain
            }], columns=features_no)
            
            # Predict the fair baseline salary (Male standard)
            base_pred_salary = model_fair.predict(profile_base)[0]
            
            # Apply job level specific OLS gender penalty dynamically to get Female salary
            penalty = job_gaps.get(sim_level, 5000.0)
            
            pred_male = base_pred_salary
            pred_female = base_pred_salary - penalty
            
            discount_inr = penalty
            discount_pct = (discount_inr / pred_male) * 100
            
            st.markdown('<h3 style="margin-top:0;">📊 AI Predictions</h3>', unsafe_allow_html=True)
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-top:4px solid #0ea5e9; text-align:center;">
                        <span class="metric-label">👨 Male Prediction</span>
                        <div class="metric-value">₹{pred_male:,.0f}</div>
                        <span style="font-size:0.8rem; color:#6b7280;">Predicted monthly salary</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_p2:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-top:4px solid #f59e0b; text-align:center;">
                        <span class="metric-label">👩 Female Prediction</span>
                        <div class="metric-value">₹{pred_female:,.0f}</div>
                        <span style="font-size:0.8rem; color:#6b7280;">Predicted monthly salary</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            st.markdown(
                f"""
                <div class="glass-card metric-badge-gap">
                    <div class="metric-label">Calculated Gender Discount</div>
                    <div class="metric-value" style="color:#f59e0b;">₹{discount_inr:,.0f} <span style="font-size:1.2rem; font-weight:400; color:#9ca3af;">({discount_pct:.1f}% discount)</span></div>
                    <p class="metric-desc" style="margin-bottom:0;">
                        For identical credentials, a female worker faces an unexplained OLS pay penalty of <strong>₹{discount_inr:,.0f} less per month</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.write("##### 📊 Salary Predictors: Feature Importance")
            importance_df = get_feature_importances(model_fair, features_no)
            fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#0ea5e9'])
            fig_imp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f3f4f6'), xaxis=dict(title="Importance Weight", gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(title=""), margin=dict(t=10, b=10, l=10, r=10), height=230
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # ── TAB 2: PERSONAL FAIR PAY AUDIT ──────────────────────────────────────────
    with tab2:
        st.subheader("Personal Fair Pay Audit")
        st.write(
            "Input your actual monthly salary and credentials. The AI compares your compensation to the male market baseline "
            "(counterfactual prediction) to audit if you face gender-based underpayment."
        )
        
        left_aud, right_aud = st.columns([1, 1])
        
        with left_aud:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("#### 💼 Audit Parameters")
            aud_salary = st.number_input("Your Actual Monthly Salary (INR)", min_value=10000, max_value=500000, value=75000, step=5000, key="aud_sal_in")
            
            aud_exp = st.slider("Your Experience (Years)", min_value=0.0, max_value=25.0, value=8.0, step=0.5, key="aud_exp_in")
            aud_age = st.slider("Your Age (Years)", min_value=20, max_value=60, value=30, key="aud_age_in")
            aud_level = st.selectbox("Your Job Level", options=['Junior', 'Mid', 'Senior', 'Manager', 'Director'], index=2, key="aud_lvl_in")
            aud_industry = st.selectbox("Your Industry Sector", options=sorted(df['industry'].unique()), index=8, key="aud_ind_in")
            aud_domain = st.selectbox("Your Job Domain", options=sorted(df['domain'].unique()), index=2, key="aud_dom_in")
            aud_education = st.selectbox("Your Education Level", options=sorted(df['education_level'].unique()), index=1, key="aud_edu_in")
            aud_company = st.selectbox("Your Company Size", options=sorted(df['company_size'].unique()), key="aud_size_in")
            aud_city = st.selectbox("Your City Tier", options=sorted(df['city_tier'].unique()), key="aud_city_in")
            aud_rating = st.slider("Your last Performance Rating (1-5)", min_value=1, max_value=5, value=4, key="aud_rate_in")
            
            aud_neg = st.checkbox("Did you negotiate your salary?", value=False, key="aud_neg_in")
            aud_ot = st.checkbox("Do you work overtime regularly?", value=False, key="aud_ot_in")
            aud_care = st.checkbox("Are you a primary caregiver?", value=False, key="aud_care_in")
            aud_reloc = st.checkbox("Did you relocate for this job?", value=False, key="aud_reloc_in")
            aud_gap = st.number_input("Career break (months)", min_value=0, max_value=48, value=0, key="aud_gap_in")
            aud_promo = st.number_input("Number of promotions", min_value=0, max_value=6, value=2, key="aud_promo_in")
            
            aud_neg_val = 1 if aud_neg else 0
            aud_ot_val = 1 if aud_ot else 0
            aud_care_val = 1 if aud_care else 0
            aud_reloc_val = 1 if aud_reloc else 0
            st.markdown('</div>', unsafe_allow_html=True)
            
        with right_aud:
            # Predict merit salary using male benchmark model
            profile_aud = pd.DataFrame([{
                'years_of_experience': aud_exp, 'job_level': aud_level, 'industry': aud_industry, 
                'company_size': aud_company, 'negotiated_salary': aud_neg_val, 'employment_type': 'Full-Time', 
                'career_gap_months': aud_gap, 'num_promotions': aud_promo, 'primary_caregiver': aud_care_val, 
                'relocated_for_job': aud_reloc_val, 'works_overtime': aud_ot_val, 'education_level': aud_education, 
                'city_tier': aud_city, 'performance_rating': aud_rating, 'age': aud_age, 'domain': aud_domain
            }], columns=features_no)
            
            fair_pred = model_fair.predict(profile_aud)[0]
            diff = aud_salary - fair_pred
            diff_pct = (diff / fair_pred) * 100
            
            st.markdown('<h3 style="margin-top:0;">🔍 Audit Assessment</h3>', unsafe_allow_html=True)
            
            if diff_pct < -5.0:
                badge_style = "metric-badge-gap"
                text_color = "#f59e0b"
                assessment = "⚠️ Underpaid"
                desc = (
                    f"Our AI predicts that your credentials command an unbiased market salary of "
                    f"<strong>₹{fair_pred:,.0f} per month</strong>. You are currently earning "
                    f"<strong style='color:#f59e0b;'>{abs(diff_pct):.1f}% less</strong> than the male market benchmark for your credentials. "
                    "Consider referencing these objective numbers during salary review meetings."
                )
            elif diff_pct > 5.0:
                badge_style = "metric-badge-fair"
                text_color = "#10b981"
                assessment = "✨ Well Compensated"
                desc = (
                    f"Our AI predicts a market salary baseline of <strong>₹{fair_pred:,.0f} per month</strong>. "
                    f"You are earning <strong style='color:#10b981;'>{diff_pct:.1f}% more</strong> than standard market averages. "
                    "Your compensation is highly competitive!"
                )
            else:
                badge_style = "metric-badge-info"
                text_color = "#0ea5e9"
                assessment = "🤝 Fairly Paid"
                desc = (
                    f"Our AI predicts a market salary baseline of <strong>₹{fair_pred:,.0f} per month</strong>. "
                    f"Your actual salary of <strong>₹{aud_salary:,.0f}</strong> aligns closely with market standards."
                )
                
            st.markdown(
                f"""
                <div class="glass-card {badge_style}">
                    <span class="metric-label">Audit Assessment</span>
                    <div class="metric-value" style="color:{text_color};">{assessment}</div>
                    <p style="font-size:0.95rem; color:#d1d5db; line-height:1.6; margin-top:8px;">
                        {desc}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"""
                <div class="glass-card">
                    <h5 style="margin-top:0; color:#10b981;">💡 Market Benchmark Table</h5>
                    <table style="width:100%; font-size:0.9rem; color:#d1d5db; border-collapse: collapse;">
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05); height:30px;">
                            <td>Your actual monthly salary:</td>
                            <td style="text-align:right; font-weight:700;">₹{aud_salary:,.0f}</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05); height:30px;">
                            <td>Merit-based market prediction:</td>
                            <td style="text-align:right; font-weight:700;">₹{fair_pred:,.0f}</td>
                        </tr>
                        <tr style="height:35px;">
                            <td>Compensation Discrepancy:</td>
                            <td style="text-align:right; font-weight:700; color:{text_color};">₹{diff:,.0f} ({diff_pct:+.1f}%)</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ── TAB 3: EQUISIM POLICY SANDBOX ──────────────────────────────────────────
    with tab3:
        st.subheader("EquiSim Policy Simulator Sandbox")
        st.write(
            "Adjust the sliders below to introduce corporate equity policies. The simulator will "
            "apply these policies to the employee database and run predictions to show how the overall pay gap decreases."
        )
        
        col_sliders, col_results = st.columns([1.1, 1])
        
        with col_sliders:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("### 🛠️ Policy Interventions")
            
            current_female_neg = df[df['gender'] == 'Female']['negotiated_salary'].mean() * 100
            st.write(f"**Policy 1: Negotiation Equity** (Current Female Rate: {current_female_neg:.1f}%)")
            sim_neg_rate = st.slider(
                "Target Negotiation % for Female Employees",
                min_value=int(current_female_neg),
                max_value=100,
                value=int(current_female_neg) + 20,
                step=5,
                key="p4_sim_neg"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.write("**Policy 2: Returnship & Childcare Support**")
            sim_gap_reduction = st.slider(
                "Reduction in Career Break Months for Women (%)",
                min_value=0,
                max_value=80,
                value=20,
                step=5,
                key="p4_sim_gap"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            current_female_promo = df[df['gender'] == 'Female']['num_promotions'].mean()
            current_male_promo = df[df['gender'] == 'Male']['num_promotions'].mean()
            st.write(f"**Policy 3: Equal Promotions** (Current Female: {current_female_promo:.2f} vs Male: {current_male_promo:.2f})")
            sim_promo_increase = st.slider(
                "Increase Female Promotion Average by (%)",
                min_value=0,
                max_value=100,
                value=30,
                step=5,
                key="p4_sim_promo"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_results:
            sim_df = df.copy()
            female_mask = sim_df['gender'] == 'Female'
            female_no_neg_mask = female_mask & (sim_df['negotiated_salary'] == 0)
            
            current_neg_fraction = current_female_neg / 100.0
            target_neg_fraction = sim_neg_rate / 100.0
            additional_needed = int((target_neg_fraction - current_neg_fraction) * len(df[female_mask]))
            
            if additional_needed > 0:
                no_neg_indices = sim_df[female_no_neg_mask].index.tolist()
                if no_neg_indices:
                    np.random.seed(42)
                    chosen_indices = np.random.choice(no_neg_indices, size=min(additional_needed, len(no_neg_indices)), replace=False)
                    sim_df.loc[chosen_indices, 'negotiated_salary'] = 1
                    
            sim_df.loc[female_mask, 'career_gap_months'] = sim_df.loc[female_mask, 'career_gap_months'] * (1 - sim_gap_reduction / 100.0)
            
            max_increase = current_male_promo - current_female_promo
            addition = max_increase * (sim_promo_increase / 100.0)
            sim_df.loc[female_mask, 'num_promotions'] = sim_df.loc[female_mask, 'num_promotions'] + addition
            
            # Predict salaries for all employees in the simulation
            sim_df['monthly_salary_inr'] = model_bias.predict(sim_df[features_with])
            
            sim_avg_male = sim_df[sim_df['gender'] == 'Male']['monthly_salary_inr'].mean()
            sim_avg_female = sim_df[sim_df['gender'] == 'Female']['monthly_salary_inr'].mean()
            sim_pay_gap = ((sim_avg_male - sim_avg_female) / sim_avg_male) * 100
            
            # Predict baseline salaries using the model for a mathematically consistent comparison
            baseline_salaries = model_bias.predict(df[features_with])
            current_avg_male = baseline_salaries[df['gender'] == 'Male'].mean()
            current_avg_female = baseline_salaries[df['gender'] == 'Female'].mean()
            current_pay_gap = ((current_avg_male - current_avg_female) / current_avg_male) * 100
            
            gap_reduction = current_pay_gap - sim_pay_gap
            
            # Calculate female wage delta for highlight
            female_delta = sim_avg_female - current_avg_female
            female_delta_text = f" <span style='color:#10b981; font-size:0.8rem; font-weight:bold;'>(+₹{female_delta:,.0f})</span>" if female_delta > 1.0 else ""
            
            st.markdown('<h3 style="margin-top:0;">📊 Simulation Output</h3>', unsafe_allow_html=True)
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-top:4px solid #f59e0b; text-align:center; padding-bottom: 20px;">
                        <span class="metric-label">Current Pay Gap</span>
                        <div class="metric-value" style="color:#f59e0b; margin-bottom:12px;">{current_pay_gap:.2f}%</div>
                        <div style="font-size:0.85rem; color:#9ca3af; text-align:left; padding: 0 12px; border-top: 1px solid rgba(255,255,255,0.05); padding-top:10px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                                <span>👨 Avg Male:</span>
                                <strong style="color:#f3f4f6;">₹{current_avg_male:,.0f}</strong>
                            </div>
                            <div style="display:flex; justify-content:space-between;">
                                <span>👩 Avg Female:</span>
                                <strong style="color:#f3f4f6;">₹{current_avg_female:,.0f}</strong>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_res2:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-top:4px solid #10b981; text-align:center; padding-bottom: 20px;">
                        <span class="metric-label">Simulated Pay Gap</span>
                        <div class="metric-value" style="color:#10b981; margin-bottom:12px;">{sim_pay_gap:.2f}%</div>
                        <div style="font-size:0.85rem; color:#9ca3af; text-align:left; padding: 0 12px; border-top: 1px solid rgba(255,255,255,0.05); padding-top:10px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                                <span>👨 Avg Male:</span>
                                <strong style="color:#f3f4f6;">₹{sim_avg_male:,.0f}</strong>
                            </div>
                            <div style="display:flex; justify-content:space-between;">
                                <span>👩 Avg Female:</span>
                                <strong style="color:#f3f4f6;">₹{sim_avg_female:,.0f}{female_delta_text}</strong>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            if gap_reduction >= -0.01:
                badge_style = "metric-badge-fair"
                text_color = "#10b981"
                action_text = f"narrowed the pay gap from <strong>{current_pay_gap:.1f}%</strong> to <strong>{sim_pay_gap:.1f}%</strong>!"
                metric_unit = "reduction"
                display_val = max(0.0, gap_reduction)
            else:
                badge_style = "metric-badge-gap"
                text_color = "#f59e0b"
                action_text = f"widened the pay gap from <strong>{current_pay_gap:.1f}%</strong> to <strong>{sim_pay_gap:.1f}%</strong> due to below-baseline parameters."
                metric_unit = "increase"
                display_val = abs(gap_reduction)

            st.markdown(
                f"""
                <div class="glass-card {badge_style}">
                    <div class="metric-label">Gap Reduction Achieved</div>
                    <div class="metric-value" style="color:{text_color};">{display_val:.2f}% <span style="font-size:1.1rem; font-weight:400; color:#9ca3af;">{metric_unit}</span></div>
                    <p class="metric-desc" style="margin-bottom:0;">
                        Your policies {action_text}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Simple chart
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Bar(
                x=['Current Gap', 'Simulated Gap'],
                y=[current_pay_gap, sim_pay_gap],
                marker_color=['#f59e0b', '#10b981'],
                width=0.4
            ))
            fig_sim.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f3f4f6'), yaxis=dict(title="Pay Gap (%)", gridcolor='rgba(255,255,255,0.05)', range=[0, 35]),
                margin=dict(t=10, b=10, l=10, r=10), height=180
            )
            st.plotly_chart(fig_sim, use_container_width=True)

    # ── TAB 4: CORPORATE STRATEGY GENERATOR ──────────────────────────────────────
    with tab4:
        st.subheader("Corporate Equity Strategy Generator")
        st.write("Select an industry below to generate a tailored strategic action plan for HR managers.")
        
        strategy_industry = st.selectbox("Select Target Industry", options=sorted(df['industry'].unique()), index=8, key="strat_ind_box")
        
        ind_df = df[df['industry'] == strategy_industry]
        ind_m_sal = ind_df[ind_df['gender'] == 'Male']['monthly_salary_inr'].mean()
        ind_f_sal = ind_df[ind_df['gender'] == 'Female']['monthly_salary_inr'].mean()
        ind_gap = ((ind_m_sal - ind_f_sal) / ind_m_sal) * 100
        
        st.write(f"For the **{strategy_industry}** sector, the current gender pay gap stands at **{ind_gap:.1f}%**.")
        
        if strategy_industry in ['Technology', 'Consulting', 'Finance & Banking']:
            st.markdown(
                """
                <div class="highlight-box">
                    <h4 style="margin-top:0; color:#10b981;">🎯 Tailored Action Plan for High-Paying Knowledge Sectors</h4>
                    <p><strong>Primary Gap Driver:</strong> Lower negotiation rates & slower promotions to Manager/Director levels.</p>
                    <ul>
                        <li><strong>Standardized Compensation Ranges:</strong> Eliminate negotiation-based disparities by establishing transparent, narrow pay bands for each role and level. Make starting offers non-negotiable.</li>
                        <li><strong>Bias-Free Promotional Audits:</strong> Mandate representation requirements for promotions. Audit the promotion pipeline every six months to identify if women are stagnating at Senior levels.</li>
                        <li><strong>Flexible/Hybrid Working Models:</strong> Standardize work-from-home and flexible hours to support caregivers without career gaps.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif strategy_industry in ['Manufacturing', 'FMCG', 'Retail']:
            st.markdown(
                """
                <div class="highlight-box">
                    <h4 style="margin-top:0; color:#10b981;">🎯 Tailored Action Plan for Traditional & Operations Sectors</h4>
                    <p><strong>Primary Gap Driver:</strong> Overtime premium differentials & career breaks for caregivers.</p>
                    <ul>
                        <li><strong>Overtime Equity Review:</strong> Standardize how overtime work is allocated and compensated. Ensure women are not excluded from overtime opportunities due to shifts or safety regulations.</li>
                        <li><strong>On-Site Daycare & Creche Facilities:</strong> Implement state-of-the-art childcare infrastructure or subsidies to reduce maternal career breaks.</li>
                        <li><strong>Returnship Programs:</strong> Build structured 3-month returnee programs for women returning from career breaks, offering skill-refreshers and immediate permanent placement.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="highlight-box">
                    <h4 style="margin-top:0; color:#10b981;">🎯 General Equity Action Plan</h4>
                    <p><strong>Primary Gap Driver:</strong> Caregiver gaps & negotiation return differences.</p>
                    <ul>
                        <li><strong>Equal Parental Leave:</strong> Implement mandatory paternity leave policies alongside maternity leave to equalize caregiver expectations across genders.</li>
                        <li><strong>Negotiation Workshops for Women:</strong> Offer professional advocacy workshops, coupled with manager training to eliminate penalization of women who negotiate.</li>
                        <li><strong>Annual Pay Auditing:</strong> Review market compensation benchmarks annually using AI-driven toolsets to audit employee compensation based strictly on merit.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
