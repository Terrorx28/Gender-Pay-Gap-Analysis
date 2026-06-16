import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_data

def show():
    # Use session state dataset if available, otherwise load directly
    if 'df' in st.session_state:
        df = st.session_state.df
    else:
        df = load_data()
        st.session_state.df = df

    # ── Page Header ──────────────────────────────────────────────────────────
    st.markdown('<h1 class="main-title">🏢 Gender Pay Gap Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Gender Pay Gap in India: Problem Statement & Executive Summary</p>', unsafe_allow_html=True)

    # ── KPI cards ────────────────────────────────────────────────────────────
    male_avg   = df[df["gender"] == "Male"]["monthly_salary_inr"].mean()
    female_avg = df[df["gender"] == "Female"]["monthly_salary_inr"].mean()
    gap_pct    = (male_avg - female_avg) / male_avg * 100
    gap_abs    = male_avg - female_avg
    female_pct = df[df["gender"] == "Female"].shape[0] / len(df) * 100

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="glass-card metric-badge-info">
                <div class="metric-label">Avg Male Salary</div>
                <div class="metric-value">₹{male_avg:,.0f}</div>
                <p class="metric-desc">Average monthly salary for male employees.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""
            <div class="glass-card metric-badge-fair">
                <div class="metric-label">Avg Female Salary</div>
                <div class="metric-value">₹{female_avg:,.0f}</div>
                <p class="metric-desc">Average monthly salary for female employees.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""
            <div class="glass-card metric-badge-gap">
                <div class="metric-label">Raw Pay Gap</div>
                <div class="metric-value" style="color:#f59e0b;">{gap_pct:.1f}%</div>
                <p class="metric-desc">Difference in average monthly salaries (Men earn more).</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col4:
        st.markdown(
            f"""
            <div class="glass-card metric-badge-info">
                <div class="metric-label">Women in Dataset</div>
                <div class="metric-value">{female_pct:.1f}%</div>
                <p class="metric-desc">Female representation in the 2,002 employee workforce.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Problem Statement ────────────────────────────────────────────────────
    st.subheader("🎯 Problem Statement")

    st.markdown(
        """
        <div class="highlight-box">
            <h4 style="margin-top:0; color:#10b981;">📌 The Core Issue</h4>
            <p style="margin-bottom:0; font-size:0.95rem; color:#e5e7eb;">
                Despite equal-work-equal-pay laws in India (Equal Remuneration Act, 1976),
                women consistently earn less than men across virtually every sector.
                This project quantifies <strong>how large</strong> the gap is,
                <strong>where</strong> it exists, and <strong>why</strong> it persists —
                using machine learning to separate structural discrimination from
                legitimate salary drivers.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown("""
        #### **Key Questions We Answer:**
        * 🔴 What is the raw gender pay gap across industries & job levels?
        * 🟠 How much of the gap is explained by experience, education, promotions?
        * 🟡 What is the **unexplained** gap — a proxy for discrimination?
        * 🟢 Which factors predict salary best — and does gender matter independently?
        * 🔵 Can a model trained without gender still predict salaries accurately?
        
        #### **Why It Matters:**
        > India ranks **127th out of 146 countries** in the Global Gender Gap Index
        > (World Economic Forum). The gender wage gap is a major driver.
        > Closing it could add **$770 billion** to India's GDP annually (McKinsey).
        """)

    with col_r:
        # Interactive Plotly Pie Chart for Gender Split
        gender_counts = df["gender"].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=gender_counts.index,
            values=gender_counts.values,
            hole=0.55,
            marker=dict(colors=['#0ea5e9', '#f59e0b']),
            textinfo='percent+label',
            textfont=dict(size=12, color='#ffffff'),
            hoverinfo='label+value+percent'
        )])
        
        fig_pie.update_layout(
            title=dict(text="Gender Distribution in Dataset", font=dict(size=15, family="Outfit")),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f3f4f6'),
            showlegend=False,
            margin=dict(t=50, b=10, l=10, r=10),
            height=280
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Dataset Overview ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📁 About the Dataset")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **👥 Demographics**
        - 2,002 employees
        - Male: 1,100 | Female: 902
        - Age range: 22–48 years
        - Experience: 0.5–26.5 years
        """)
    with col_b:
        st.markdown("""
        **🏢 Work Context**
        - 10 Industries (Tech, Finance, FMCG…)
        - 10 Functional Domains
        - 5 Job Levels (Junior → Director)
        - 4 Company Sizes
        """)
    with col_c:
        st.markdown("""
        **📋 Key Variables**
        - Monthly salary (INR)
        - Education level
        - Performance rating
        - Promotions, career gaps
        - Negotiation, overtime, relocation
        """)

    # ── Salary distribution teaser ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 First Look: Salary Distributions")

    # Plotly overlapping histogram
    fig_hist = px.histogram(
        df,
        x="monthly_salary_inr",
        color="gender",
        barmode="overlay",
        nbins=50,
        color_discrete_map={'Male': '#0ea5e9', 'Female': '#f59e0b'},
        opacity=0.6
    )
    
    # Add vertical mean lines
    fig_hist.add_vline(x=male_avg, line_width=2, line_dash="dash", line_color="#0ea5e9", 
                       annotation_text=f"Male Mean: ₹{male_avg:,.0f}", annotation_position="top right")
    fig_hist.add_vline(x=female_avg, line_width=2, line_dash="dash", line_color="#f59e0b", 
                       annotation_text=f"Female Mean: ₹{female_avg:,.0f}", annotation_position="top left")

    fig_hist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f3f4f6'),
        xaxis=dict(title="Monthly Salary (INR)", gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title="Number of Employees", gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(t=20, b=20, l=10, r=10),
        height=320,
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right")
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(
        f"""
        <div class="glass-card" style="margin-top:15px;">
            <h4 style="margin-top:0; color:#10b981;">🔍 What the Distribution Tells Us</h4>
            <p style="margin-bottom:0; font-size:0.95rem; color:#d1d5db;">
                The male salary distribution is right-skewed — more men occupy the high-salary range
                (₹1,50,000+). Women cluster between ₹30,000–₹90,000.
                The mean gap is <strong>₹{gap_abs:,.0f}/month (≈ {gap_pct:.1f}%)</strong>.
                This gap only grows when we control for experience and job level — navigate to
                <em>Root Cause Explorer</em> to see why.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Raw data explorer ────────────────────────────────────────────────────
    with st.expander("🗂️ Browse Raw Dataset"):
        st.markdown("Showing first 50 rows. Use the filters below to explore.")
        gender_f = st.multiselect("Filter by Gender", ["Male", "Female"], default=["Male", "Female"],
                                   key="intro_gender")
        industry_f = st.multiselect("Filter by Industry", sorted(df["industry"].unique()),
                                     default=sorted(df["industry"].unique())[:3], key="intro_industry")
        view = df[df["gender"].isin(gender_f) & df["industry"].isin(industry_f)].head(50)
        st.dataframe(view.reset_index(drop=True), use_container_width=True, height=250)
        st.caption(f"Showing {len(view)} rows out of {len(df)}")

    st.markdown('<p style="text-align:center;color:#4b5563;font-size:0.75rem;margin-top:30px;">Gender Pay Gap Analysis · 2026</p>',
                unsafe_allow_html=True)