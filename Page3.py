import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.data_loader import load_data

# Train ML Models - exported for Page4
@st.cache_resource
def train_models(df):
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
    
    # Model Pipelines
    pipeline_with_gender = Pipeline(steps=[
        ('preprocessor', preprocessor_with_gender),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    pipeline_no_gender = Pipeline(steps=[
        ('preprocessor', preprocessor_no_gender),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    X_with = df[features_with_gender]
    y_with = df['monthly_salary_inr']
    pipeline_with_gender.fit(X_with, y_with)
    
    df_males = df[df['gender'] == 'Male']
    X_no = df_males[features_no_gender]
    y_no = df_males['monthly_salary_inr']
    pipeline_no_gender.fit(X_no, y_no)
    
    return pipeline_with_gender, pipeline_no_gender, features_with_gender, features_no_gender

def show():
    # Use session state dataset if available
    if 'df' in st.session_state:
        df = st.session_state.df
    else:
        df = load_data()
        st.session_state.df = df

    st.markdown('<h1 class="main-title">🔍 Root Cause Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Decompose the Gender Pay Gap and Quantify the Motherhood & Negotiation Penalty</p>', unsafe_allow_html=True)

    # ── SECTION 1 — Blinder-Oaxaca pooled OLS decomposition ──────────────────────
    st.markdown('<div class="section-header" style="font-size:1.5rem; margin-top:20px; color:#10b981;">🧮 Gap Decomposition: Qualifications vs. Direct Bias</div>', unsafe_allow_html=True)

    st.markdown("""
    To understand what drives the gender pay gap, we use the **Blinder-Oaxaca pooled OLS decomposition**. 
    This technique splits the raw wage gap into two parts:
    1. **Explained Gap (Qualifications & Role):** The portion due to differences in credentials (experience, job level seniority, and education representation).
    2. **Unexplained Gap (Direct Bias):** The remaining salary disparity for employees with identical credentials — representing the direct equal-pay-for-equal-work penalty.
    """)

    # Raw averages
    male_mean   = df[df["gender"] == "Male"]["monthly_salary_inr"].mean()
    female_mean = df[df["gender"] == "Female"]["monthly_salary_inr"].mean()
    raw_gap     = male_mean - female_mean

    # Fit OLS on merit features to calculate exact decomposition components
    merit_df = df[['gender', 'years_of_experience', 'job_level', 'education_level', 'monthly_salary_inr']].copy()
    merit_encoded = pd.get_dummies(merit_df, drop_first=True)
    
    X = merit_encoded.drop(columns=['monthly_salary_inr'])
    y = merit_encoded['monthly_salary_inr']
    lr = LinearRegression().fit(X, y)
    
    gender_col = [c for c in X.columns if 'gender_Male' in c][0]
    unexplained = lr.coef_[X.columns.get_loc(gender_col)]
    explained = raw_gap - unexplained

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="glass-card metric-badge-info">
            <div class="metric-label">Raw Pay Gap</div>
            <div class="metric-value">₹{raw_gap:,.0f}</div>
            <p class="metric-desc">Overall average salary gap per month.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="glass-card metric-badge-fair">
            <div class="metric-label">Explained by Seniority & Role</div>
            <div class="metric-value">₹{explained:,.0f}</div>
            <p class="metric-desc"><strong>{explained/raw_gap*100:.1f}%</strong> of gap due to career choices & glass ceiling barriers.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="glass-card metric-badge-gap">
            <div class="metric-label">Unexplained Gap (Direct Bias)</div>
            <div class="metric-value" style="color:#f59e0b;">₹{unexplained:,.0f}</div>
            <p class="metric-desc"><strong>{unexplained/raw_gap*100:.1f}%</strong> of gap due to direct wage discrimination.</p>
        </div>""", unsafe_allow_html=True)

    # Plotly Waterfall Chart representing the decomposition
    fig_waterfall = go.Figure(go.Waterfall(
        name="Oaxaca Decomposition", 
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Male Avg Salary", "Explained Gap (Seniority/Role)", "Unexplained Gap (Direct Bias)", "Female Avg Salary"],
        textposition="outside",
        text=[f"₹{male_mean:,.0f}", f"-₹{explained:,.0f}", f"-₹{unexplained:,.0f}", f"₹{female_mean:,.0f}"],
        y=[male_mean, -explained, -unexplained, female_mean],
        connector=dict(line=dict(color="rgba(255, 255, 255, 0.2)")),
        decreasing=dict(marker=dict(color="#f59e0b")),
        increasing=dict(marker=dict(color="#10b981")),
        totals=dict(marker=dict(color="#0ea5e9"))
    ))
    
    fig_waterfall.update_layout(
        title=dict(text="Blinder-Oaxaca Wage Decomposition Waterfall", font=dict(size=14, family="Outfit")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f3f4f6'),
        yaxis=dict(title="Monthly Salary (INR)", gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(t=50, b=20, l=10, r=10),
        height=320
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.markdown(f"""
    <div class="highlight-box">
        <h4 style="margin-top:0; color:#10b981;">💡 Wage Gap Deconstruction Insights</h4>
        <ul>
            <li><strong>Explained Gap (₹{explained:,.0f} / {explained/raw_gap*100:.1f}%):</strong> This is the major driver. Because men are promoted to high-paying leadership bands (Manager/Director) at much higher rates than women, role seniority imbalances explain most of the raw pay gap.</li>
            <li><strong>Unexplained Gap (₹{unexplained:,.0f} / {unexplained/raw_gap*100:.1f}%):</strong> Even when controlling for experience, education, and seniority grade, women face a direct pay penalty of ₹{unexplained:,.0f} less per month. This residual is a statistical proxy for direct compensation bias.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 2 — Negotiation Bias ──────────────────────────────────────────
    st.markdown('<div class="section-header" style="font-size:1.5rem; margin-top:30px; color:#10b981;">🤝 The Negotiation Gap</div>', unsafe_allow_html=True)

    male_neg_rate   = df[df["gender"] == "Male"]["negotiated_salary"].mean() * 100
    female_neg_rate = df[df["gender"] == "Female"]["negotiated_salary"].mean() * 100

    col_n1, col_n2 = st.columns([1.3, 1])
    with col_n1:
        neg_data = df.groupby(['gender', 'negotiated_salary'])['monthly_salary_inr'].mean().reset_index()
        neg_data['negotiated_salary'] = neg_data['negotiated_salary'].map({0: "Didn't Negotiate", 1: "Negotiated"})
        
        fig_neg = px.bar(
            neg_data,
            x="negotiated_salary",
            y="monthly_salary_inr",
            color="gender",
            barmode="group",
            color_discrete_map={'Male': '#0ea5e9', 'Female': '#f59e0b'}
        )
        fig_neg.update_layout(
            title=dict(text="Average Salary: Negotiated vs. Non-Negotiators", font=dict(size=13, family="Outfit")),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f3f4f6'),
            xaxis_title="",
            yaxis_title="Average Salary (INR)",
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
            margin=dict(t=40, b=10, l=10, r=10),
            height=280
        )
        st.plotly_chart(fig_neg, use_container_width=True)

    with col_n2:
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:15px;">
            <h4 style="margin-top:0; color:#f59e0b;">💬 Factors in Negotiation Rates</h4>
            <p style="font-size:0.88rem; color:#d1d5db; line-height:1.5; margin-bottom:0;">
                • <strong>Backlash Penalty:</strong> Social expectations lead to women being evaluated more negatively when initiating assertiveness.<br>
                • <strong>Information Asymmetry:</strong> Lack of peer benchmarks leaves women at a disadvantage during offers.<br>
                • <strong>Confidence Standards:</strong> Disparities in self-assessment lead to lower salary requests.<br>
                • <strong>Cultural Barriers:</strong> Social norms sometimes frame active money negotiations as less feminine.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Male Negotiation Rate", f"{male_neg_rate:.1f}%")
        with col_m2:
            st.metric("Female Negotiation Rate", f"{female_neg_rate:.1f}%",
                      delta=f"{female_neg_rate - male_neg_rate:.1f}%", delta_color="inverse")

    # ── SECTION 3 — Career Gap & Caregiver Penalty ──────────────────────────────
    st.markdown('<div class="section-header" style="font-size:1.5rem; margin-top:30px; color:#10b981;">👶 The Caregiver Penalty & Career Gaps</div>', unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_gap = px.histogram(
            df,
            x="career_gap_months",
            color="gender",
            barmode="overlay",
            nbins=20,
            color_discrete_map={'Male': '#0ea5e9', 'Female': '#f59e0b'},
            opacity=0.6
        )
        fig_gap.update_layout(
            title=dict(text="Career Break Length Distribution (Months)", font=dict(size=13, family="Outfit")),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f3f4f6'),
            xaxis=dict(title="Career Gap (months)", gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title="Count", gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
            margin=dict(t=40, b=10, l=10, r=10),
            height=280
        )
        st.plotly_chart(fig_gap, use_container_width=True)

    with col_c2:
        care_stats = df.groupby(["primary_caregiver", "gender"])["monthly_salary_inr"].mean().reset_index()
        care_stats['primary_caregiver'] = care_stats['primary_caregiver'].map({0: "Not Caregiver", 1: "Caregiver"})
        
        fig_care = px.bar(
            care_stats,
            x="primary_caregiver",
            y="monthly_salary_inr",
            color="gender",
            barmode="group",
            color_discrete_map={'Male': '#0ea5e9', 'Female': '#f59e0b'}
        )
        fig_care.update_layout(
            title=dict(text="Caregiver Status vs. Average Monthly Salary", font=dict(size=13, family="Outfit")),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f3f4f6'),
            xaxis_title="",
            yaxis_title="Average Salary (INR)",
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
            margin=dict(t=40, b=10, l=10, r=10),
            height=280
        )
        st.plotly_chart(fig_care, use_container_width=True)

    male_care   = df[(df["gender"]=="Male")   & (df["primary_caregiver"]==1)]["monthly_salary_inr"].mean()
    male_nocare = df[(df["gender"]=="Male")   & (df["primary_caregiver"]==0)]["monthly_salary_inr"].mean()
    fem_care    = df[(df["gender"]=="Female") & (df["primary_caregiver"]==1)]["monthly_salary_inr"].mean()
    fem_nocare  = df[(df["gender"]=="Female") & (df["primary_caregiver"]==0)]["monthly_salary_inr"].mean()

    st.markdown(f"""
    <div class="insight-box danger">
        <h4>👶 The Motherhood / Caregiver Penalty</h4>
        <p>
            • <strong>Men who are primary caregivers</strong> earn about ₹{male_nocare-male_care:,.0f}/month less.<br>
            • <strong>Women who are primary caregivers</strong> earn about ₹{fem_nocare-fem_care:,.0f}/month less.<br>
            • This is compounded by women taking longer breaks (averaging 7.7 months vs 2.4 months) which cuts into promotions and compound seniority growth.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 4 — Interactive Factor Explorer ─────────────────────────────────
    st.markdown('<div class="section-header" style="font-size:1.5rem; margin-top:30px; color:#10b981;">🎛️ Interactive Factor Explorer</div>', unsafe_allow_html=True)
    st.markdown("Select any categorical factor below and inspect how average salaries and gender gap percentages vary across classes.")

    factor = st.selectbox(
        "Choose a factor to explore",
        ["education_level", "company_size", "employment_type", "works_overtime", "relocated_for_job", "num_promotions"],
        format_func=lambda x: {
            "education_level": "Education Level",
            "company_size": "Company Size",
            "employment_type": "Employment Type",
            "works_overtime": "Works Overtime",
            "relocated_for_job": "Relocated for Job",
            "num_promotions": "Number of Promotions",
        }.get(x, x)
    )

    factor_stats = (
        df.groupby([factor, "gender"])["monthly_salary_inr"]
        .mean().unstack("gender").fillna(0)
    )
    if "Male" in factor_stats.columns and "Female" in factor_stats.columns:
        factor_stats["gap_pct"] = (
            (factor_stats["Male"] - factor_stats["Female"]) / factor_stats["Male"] * 100
        ).round(1)

    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        f_stats_reset = df.groupby([factor, "gender"])["monthly_salary_inr"].mean().reset_index()
        fig_f_sal = px.bar(
            f_stats_reset,
            x=factor,
            y="monthly_salary_inr",
            color="gender",
            barmode="group",
            color_discrete_map={'Male': '#0ea5e9', 'Female': '#f59e0b'}
        )
        fig_f_sal.update_layout(
            title=dict(text=f"Salary Comparison by {factor.replace('_',' ').title()}", font=dict(size=13, family="Outfit")),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f3f4f6'),
            xaxis_title="",
            yaxis_title="Average Salary (INR)",
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
            margin=dict(t=40, b=10, l=10, r=10),
            height=300
        )
        st.plotly_chart(fig_f_sal, use_container_width=True)
        
    with col_f2:
        if "gap_pct" in factor_stats.columns:
            f_gap_reset = factor_stats.reset_index()
            fig_f_gap = px.bar(
                f_gap_reset,
                x=factor,
                y="gap_pct",
                color="gap_pct",
                color_continuous_scale=["#10b981", "#f59e0b", "#f43f5e"]
            )
            fig_f_gap.update_layout(
                title=dict(text="Gender Pay Gap Percentage (%)", font=dict(size=13, family="Outfit")),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f3f4f6'),
                xaxis_title="",
                yaxis_title="Pay Gap (%)",
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                coloraxis_showscale=False,
                margin=dict(t=40, b=10, l=10, r=10),
                height=300
            )
            st.plotly_chart(fig_f_gap, use_container_width=True)

    with st.expander("📋 View Detailed Summary Table"):
        styler = factor_stats.rename(
            columns={"Male": "Male Avg Salary (INR)", "Female": "Female Avg Salary (INR)", "gap_pct": "Gender Pay Gap (%)"}
        ).style
        
        def color_gap(val):
            try:
                v = float(val)
                if v > 15:
                    return 'background-color: rgba(244, 63, 94, 0.15); color: #f43f5e; font-weight: bold;'
                elif v > 5:
                    return 'background-color: rgba(245, 158, 11, 0.15); color: #f59e0b; font-weight: bold;'
                else:
                    return 'background-color: rgba(16, 185, 129, 0.15); color: #10b981; font-weight: bold;'
            except Exception:
                return ''
                
        if hasattr(styler, 'map'):
            styled_df = styler.map(color_gap, subset=["Gender Pay Gap (%)"])
        else:
            styled_df = styler.applymap(color_gap, subset=["Gender Pay Gap (%)"])
            
        st.dataframe(styled_df, use_container_width=True)

    st.markdown('<p style="text-align:center;color:#4b5563;font-size:0.75rem;margin-top:30px;">Gender Pay Gap Analysis · Page 3 · Root Cause Explorer</p>',
                unsafe_allow_html=True)