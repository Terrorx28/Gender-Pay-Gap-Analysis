import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

def show():
    # Page Header
    st.markdown('<h1 class="main-title">📊 Disparity Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Interactive Deep-Dive into Salary Distributions, Career Gaps, and Regression Models</p>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.error("Data could not be loaded. Please reload the main page.")
        return
        
    df = st.session_state.df
    
    # Expandable Filter Panel
    with st.expander("🔍 Interactive Data Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_industries = st.multiselect(
                "Filter by Industry", 
                options=sorted(df['industry'].unique()),
                default=[]
            )
            selected_domains = st.multiselect(
                "Filter by Functional Domain", 
                options=sorted(df['domain'].unique()),
                default=[]
            )
        with col2:
            selected_education = st.multiselect(
                "Filter by Education Level", 
                options=sorted(df['education_level'].unique()),
                default=[]
            )
            selected_job_levels = st.multiselect(
                "Filter by Job Level", 
                options=sorted(df['job_level'].unique()),
                default=[]
            )
        with col3:
            selected_city_tiers = st.multiselect(
                "Filter by City Tier", 
                options=sorted(df['city_tier'].unique()),
                default=[]
            )
            selected_company_sizes = st.multiselect(
                "Filter by Company Size", 
                options=sorted(df['company_size'].unique()),
                default=[]
            )
            
    # Apply Filters
    df_filtered = df.copy()
    if selected_industries:
        df_filtered = df_filtered[df_filtered['industry'].isin(selected_industries)]
    if selected_domains:
        df_filtered = df_filtered[df_filtered['domain'].isin(selected_domains)]
    if selected_education:
        df_filtered = df_filtered[df_filtered['education_level'].isin(selected_education)]
    if selected_job_levels:
        df_filtered = df_filtered[df_filtered['job_level'].isin(selected_job_levels)]
    if selected_city_tiers:
        df_filtered = df_filtered[df_filtered['city_tier'].isin(selected_city_tiers)]
    if selected_company_sizes:
        df_filtered = df_filtered[df_filtered['company_size'].isin(selected_company_sizes)]
        
    if df_filtered.empty:
        st.warning("⚠️ No records match the selected filters. Please expand your filter criteria.")
        return
        
    # Stats summary for filtered data
    total_records = len(df_filtered)
    males_count = len(df_filtered[df_filtered['gender'] == 'Male'])
    females_count = len(df_filtered[df_filtered['gender'] == 'Female'])
    
    avg_male = df_filtered[df_filtered['gender'] == 'Male']['monthly_salary_inr'].mean() if males_count > 0 else 0
    avg_female = df_filtered[df_filtered['gender'] == 'Female']['monthly_salary_inr'].mean() if females_count > 0 else 0
    gap = ((avg_male - avg_female) / avg_male * 100) if avg_male > 0 else 0
    
    # Render mini stats bar
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); padding:10px 20px; border-radius:8px; margin-bottom:25px; font-size:0.9rem;">
            <span>📋 Showing <strong>{total_records}</strong> employees ({males_count} Men, {females_count} Women)</span>
            <span>👨 Avg Male: <strong>₹{avg_male:,.0f}</strong></span>
            <span>👩 Avg Female: <strong>₹{avg_female:,.0f}</strong></span>
            <span>⚖️ Filtered Raw Gap: <strong style="color:{'#f59e0b' if gap > 0 else '#10b981'};">{gap:.1f}%</strong></span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # TAB LAYOUT FOR EXPLORATION
    tab1, tab2, tab3 = st.tabs(["📊 Distribution & Paradox", "🤝 Negotiation & Overtime returns", "🧮 Econometric Bias Calculator (Regression)"])
    
    with tab1:
        col_t1_left, col_t1_right = st.columns([1, 1])
        
        with col_t1_left:
            st.subheader("Salary Spread & Distributions")
            st.write("Understand the density of pay. Note the difference in medians and the higher outliers for males.")
            
            # Violin Plot
            fig_violin = px.violin(
                df_filtered, 
                x="gender", 
                y="monthly_salary_inr", 
                color="gender", 
                box=True, 
                points="outliers",
                color_discrete_map={'Male': 'rgba(14, 165, 233, 0.8)', 'Female': 'rgba(245, 158, 11, 0.8)'}
            )
            fig_violin.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f3f4f6'),
                xaxis_title="Gender",
                yaxis_title="Monthly Salary (INR)",
                showlegend=False,
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                margin=dict(t=10, b=10, l=10, r=10),
                height=350
            )
            st.plotly_chart(fig_violin, use_container_width=True)
            
        with col_t1_right:
            st.subheader("The Experience-Pay Paradox")
            st.write("Holding job levels constant, do women get rewarded equally for their years of experience?")
            
            # Experience vs Salary scatter plot
            fig_scatter = px.scatter(
                df_filtered,
                x="years_of_experience",
                y="monthly_salary_inr",
                color="gender",
                hover_data=["job_level", "industry", "performance_rating"],
                color_discrete_map={'Male': '#0ea5e9', 'Female': '#f59e0b'},
                opacity=0.6
            )
            
            # Fit manual trend lines using scikit-learn
            for g, col in [('Male', '#0ea5e9'), ('Female', '#f59e0b')]:
                sub = df_filtered[df_filtered['gender'] == g]
                if len(sub) > 1:
                    X_fit = sub[['years_of_experience']].values
                    y_fit = sub['monthly_salary_inr'].values
                    lr_fit = LinearRegression().fit(X_fit, y_fit)
                    
                    x_range = np.linspace(sub['years_of_experience'].min(), sub['years_of_experience'].max(), 50)
                    y_pred = lr_fit.predict(x_range.reshape(-1, 1))
                    
                    fig_scatter.add_trace(go.Scatter(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        name=f'{g} Trend',
                        line=dict(color=col, width=2.5, dash='dash')
                    ))
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f3f4f6'),
                xaxis_title="Years of Experience",
                yaxis_title="Monthly Salary (INR)",
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
                margin=dict(t=10, b=10, l=10, r=10),
                height=350
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="margin-top:0; color:#10b981;">💡 Analytical Takeaway: The Experience Standard</h4>
                <p style="margin-bottom:0; font-size:0.9rem; color:#d1d5db;">
                    Look closely at the scatter plot trendlines. Notice how the trendline for female employees lies consistently below the male trendline. Within the same bands of experience, women are systematically compensated at a lower baseline. Additionally, the regression analyses show that women are held to a <strong>higher experience standard</strong>: at any given job level (Junior, Mid, Senior, Manager), women have on average 0.7 to 1.1 years <em>more</em> experience than their male counterparts, yet they take home smaller paychecks.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with tab2:
        st.subheader("Divergence in Compensation Returns")
        st.write("Do identical professional actions yield equal financial returns for men and women?")
        
        col_t2_left, col_t2_right = st.columns(2)
        
        with col_t2_left:
            st.write("#### 🤝 Salary Negotiation Payoff")
            st.write("Does negotiation pay off equally? Average salary increase from negotiation:")
            
            # Negotiation payoff chart
            neg_data = df_filtered.groupby(['gender', 'negotiated_salary'])['monthly_salary_inr'].mean().reset_index()
            neg_data['negotiated_salary'] = neg_data['negotiated_salary'].map({0: 'Did Not Negotiate', 1: 'Negotiated Salary'})
            
            fig_neg = px.bar(
                neg_data,
                x="negotiated_salary",
                y="monthly_salary_inr",
                color="gender",
                barmode="group",
                color_discrete_map={'Male': '#0ea5e9', 'Female': '#f59e0b'}
            )
            fig_neg.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f3f4f6'),
                xaxis_title="",
                yaxis_title="Average Salary (INR)",
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
                margin=dict(t=10, b=10, l=10, r=10),
                height=300
            )
            st.plotly_chart(fig_neg, use_container_width=True)
            
            # Negotiation insights
            try:
                f_no_neg = neg_data[(neg_data['gender'] == 'Female') & (neg_data['negotiated_salary'] == 'Did Not Negotiate')]['monthly_salary_inr'].values[0]
                f_neg = neg_data[(neg_data['gender'] == 'Female') & (neg_data['negotiated_salary'] == 'Negotiated Salary')]['monthly_salary_inr'].values[0]
                m_no_neg = neg_data[(neg_data['gender'] == 'Male') & (neg_data['negotiated_salary'] == 'Did Not Negotiate')]['monthly_salary_inr'].values[0]
                m_neg = neg_data[(neg_data['gender'] == 'Male') & (neg_data['negotiated_salary'] == 'Negotiated Salary')]['monthly_salary_inr'].values[0]
                
                f_inc = ((f_neg - f_no_neg) / f_no_neg) * 100
                m_inc = ((m_neg - m_no_neg) / m_no_neg) * 100
                st.write(f"📈 **Negotiation Premium:** Men gain **{m_inc:.1f}%** from negotiation (₹{m_neg-m_no_neg:,.0f}/mo), while women gain only **{f_inc:.1f}%** (₹{f_neg-f_no_neg:,.0f}/mo).")
            except IndexError:
                st.write("💡 *Please expand filter choices to view full negotiation comparisons.*")
                
        with col_t2_right:
            st.write("#### ⏰ Overtime Compensation Payoff")
            st.write("Are overtime hours rewarded equally? Average salary of overtime workers vs non-overtime:")
            
            # Overtime payoff chart
            ot_data = df_filtered.groupby(['gender', 'works_overtime'])['monthly_salary_inr'].mean().reset_index()
            ot_data['works_overtime'] = ot_data['works_overtime'].map({0: 'No Overtime', 1: 'Works Overtime'})
            
            fig_ot = px.bar(
                ot_data,
                x="works_overtime",
                y="monthly_salary_inr",
                color="gender",
                barmode="group",
                color_discrete_map={'Male': '#0ea5e9', 'Female': '#f59e0b'}
            )
            fig_ot.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f3f4f6'),
                xaxis_title="",
                yaxis_title="Average Salary (INR)",
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
                margin=dict(t=10, b=10, l=10, r=10),
                height=300
            )
            st.plotly_chart(fig_ot, use_container_width=True)
            
            # Overtime insights
            try:
                f_no_ot = ot_data[(ot_data['gender'] == 'Female') & (ot_data['works_overtime'] == 'No Overtime')]['monthly_salary_inr'].values[0]
                f_ot = ot_data[(ot_data['gender'] == 'Female') & (ot_data['works_overtime'] == 'Works Overtime')]['monthly_salary_inr'].values[0]
                m_no_ot = ot_data[(ot_data['gender'] == 'Male') & (ot_data['works_overtime'] == 'No Overtime')]['monthly_salary_inr'].values[0]
                m_ot = ot_data[(ot_data['gender'] == 'Male') & (ot_data['works_overtime'] == 'Works Overtime')]['monthly_salary_inr'].values[0]
                
                f_ot_inc = ((f_ot - f_no_ot) / f_no_ot) * 100
                m_ot_inc = ((m_ot - m_no_ot) / m_no_ot) * 100
                st.write(f"📈 **Overtime Premium:** Men gain **{m_ot_inc:.1f}%** for working overtime, while women gain only **{f_ot_inc:.1f}%**.")
            except IndexError:
                st.write("💡 *Please expand filter choices to view full overtime comparisons.*")
                
        st.markdown(
            """
            <div class="highlight-box">
                <h4 style="margin-top:0; color:#f59e0b;">⚠️ The Negotiation Return Disparity</h4>
                <p style="margin-bottom:0; font-size:0.9rem; color:#d1d5db;">
                    This chart exposes a vital truth: <strong>telling women to negotiate is not a complete solution</strong>. In this dataset, even when women negotiate, they receive less than half the financial return of men. This reflects standard organizational bias where negotiation by women can be received with friction, leading to smaller increments than those granted to men.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab3:
        st.subheader("Econometric Adjusted Gap Calculator")
        st.write(
            "In labor economics, the **Unadjusted Pay Gap** is the raw difference in averages. "
            "The **Adjusted Pay Gap** is calculated by running a regression model to control for variables (e.g. experience, role). "
            "Select control variables below to see how the 'unexplained' gap shifts in real-time."
        )
        
        # Checkboxes for control variables
        st.write("**Select Controls to Include in Regression:**")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            ctrl_exp = st.checkbox("Years of Experience", value=True)
            ctrl_level = st.checkbox("Job Level (Seniority)", value=True)
            ctrl_edu = st.checkbox("Education Level", value=False)
        with col_c2:
            ctrl_ind = st.checkbox("Industry", value=False)
            ctrl_domain = st.checkbox("Domain (Functional Area)", value=False)
            ctrl_company = st.checkbox("Company Size", value=False)
        with col_c3:
            ctrl_neg = st.checkbox("Negotiation Status", value=False)
            ctrl_breaks = st.checkbox("Career Break Months", value=False)
            ctrl_caregiver = st.checkbox("Primary Caregiver Status", value=False)
            
        # Perform OLS regression in Python
        if total_records < 10:
            st.error("Too few records filtered to run regression analysis. Please expand filters.")
        else:
            # Prepare independent variables
            features = ['gender']
            categorical_features = []
            
            if ctrl_exp:
                features.append('years_of_experience')
            if ctrl_level:
                features.append('job_level')
                categorical_features.append('job_level')
            if ctrl_edu:
                features.append('education_level')
                categorical_features.append('education_level')
            if ctrl_ind:
                features.append('industry')
                categorical_features.append('industry')
            if ctrl_domain:
                features.append('domain')
                categorical_features.append('domain')
            if ctrl_company:
                features.append('company_size')
                categorical_features.append('company_size')
            if ctrl_neg:
                features.append('negotiated_salary')
            if ctrl_breaks:
                features.append('career_gap_months')
            if ctrl_caregiver:
                features.append('primary_caregiver')
                
            # Subset dataframe
            reg_df = df_filtered[features + ['monthly_salary_inr']].copy()
            
            # Make sure we have both genders in regression
            if reg_df['gender'].nunique() < 2:
                st.error("Error: Regression requires both genders to be present in filtered dataset.")
            else:
                # Convert categorical variables to dummies
                reg_df_encoded = pd.get_dummies(reg_df, drop_first=True)
                
                # Identify X and y
                # Since gender_Male is dummy, find its name in columns
                dummy_cols = reg_df_encoded.columns
                gender_male_col = [col for col in dummy_cols if 'gender_Male' in col]
                
                if not gender_male_col:
                    st.error("Regression failed: gender column not found.")
                else:
                    gender_col_name = gender_male_col[0]
                    X = reg_df_encoded.drop(columns=['monthly_salary_inr'])
                    y = reg_df_encoded['monthly_salary_inr']
                    
                    # Fit Linear Regression
                    lr = LinearRegression()
                    lr.fit(X, y)
                    
                    # Extract the coefficient of gender_Male
                    gender_coef = lr.coef_[X.columns.get_loc(gender_col_name)]
                    
                    # Calculate R2
                    r2_score = lr.score(X, y)
                    
                    # Display regression output
                    st.markdown("#### 🧮 Model Analysis")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.markdown(
                            f"""
                            <div class="glass-card" style="text-align:center;">
                                <div class="metric-label">Adjusted Pay Penalty for Females</div>
                                <div class="metric-value" style="color:#f59e0b;">₹{abs(gender_coef):,.2f}</div>
                                <p class="metric-desc">Controlling for selected factors, a female employee earns this much <strong>less per month</strong> than an identical male employee.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col_res2:
                        st.markdown(
                            f"""
                            <div class="glass-card" style="text-align:center;">
                                <div class="metric-label">Model Explanatory Power (R-squared)</div>
                                <div class="metric-value" style="color:#0ea5e9;">{r2_score*100:.1f}%</div>
                                <p class="metric-desc">Percentage of salary variation explained by the selected control variables.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                    # Detailed Explanation
                    st.write("##### 🧐 Econometric Interpretation")
                    st.write(
                        f"When we fit the regression model with your selected variables, we control for differences in those attributes. "
                        f"The resulting pay gap is **₹{abs(gender_coef):,.0f} per month** (down from the unadjusted gap of **₹{avg_male-avg_female:,.0f}**). "
                        "This means that even when a male and female employee have the *same* experience and job level (if selected), "
                        f"a woman is paid **₹{abs(gender_coef):,.0f} less per month** purely due to unexplained systemic gender bias."
                    )
                    
                    # Display other coefficients in an expander
                    with st.expander("🔬 View Coefficients of Control Variables"):
                        coef_df = pd.DataFrame({
                            'Variable': X.columns,
                            'Coefficient (INR impact)': lr.coef_
                        }).sort_values(by='Coefficient (INR impact)', ascending=False)
                        st.dataframe(coef_df.style.format({'Coefficient (INR impact)': '₹{:,.2f}'}))