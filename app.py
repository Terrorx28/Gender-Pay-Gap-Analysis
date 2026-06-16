import streamlit as st
import pandas as pd
import numpy as np
import os

# Set page configuration - MUST BE FIRST STREAMLIT CALL
st.set_page_config(
    page_title="Gender Pay Gap Analysis | Bias Hub",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global custom CSS styling for premium look & feel
def inject_custom_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
        
        /* Font rules */
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Inter', sans-serif !important;
        }
        
        h1, h2, h3, h4, h5, h6, .stSubheader {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
        }
        
        /* Premium Gradient Header */
        .main-title {
            background: linear-gradient(135deg, #10b981, #0ea5e9, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.2rem;
            font-weight: 800;
            text-align: left;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }
        
        .sub-title {
            color: #9ca3af;
            font-size: 1.2rem;
            font-weight: 400;
            margin-bottom: 2rem;
        }
        
        /* Custom glassmorphism container */
        .glass-card {
            background: rgba(22, 34, 53, 0.4);
            border-radius: 16px;
            border: 1px solid rgba(14, 165, 233, 0.15);
            padding: 24px;
            margin-bottom: 20px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .glass-card:hover {
            border-color: rgba(14, 165, 233, 0.35);
            box-shadow: 0 12px 40px -15px rgba(14, 165, 233, 0.15);
            transform: translateY(-2px);
        }
        
        /* Custom metric badges */
        .metric-badge-gap {
            border-left: 4px solid #f59e0b !important;
        }
        
        .metric-badge-fair {
            border-left: 4px solid #10b981 !important;
        }
        
        .metric-badge-info {
            border-left: 4px solid #0ea5e9 !important;
        }
        
        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #f3f4f6;
            line-height: 1;
            margin: 8px 0;
            font-family: 'Outfit', sans-serif;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #9ca3af;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-desc {
            font-size: 0.8rem;
            color: #6b7280;
        }
        
        /* Sleek Sidebar Custom Navigation Styles */
        [data-testid="stSidebar"] {
            background-color: #080d16 !important;
            border-right: 1px solid rgba(14, 165, 233, 0.1) !important;
        }
        
        /* Style Streamlit radio group inside sidebar as buttons */
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] {
            gap: 2px !important;
        }
        
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.015) !important;
            border: 1px solid rgba(255, 255, 255, 0.04) !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
            margin-bottom: 8px !important;
            transition: all 0.2s ease-in-out !important;
            cursor: pointer !important;
        }
        
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label:hover {
            border-color: rgba(14, 165, 233, 0.25) !important;
            background: rgba(14, 165, 233, 0.04) !important;
        }
        
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label[data-checked="true"] {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(14, 165, 233, 0.15)) !important;
            border-color: rgba(14, 165, 233, 0.4) !important;
            box-shadow: 0 4px 20px -8px rgba(14, 165, 233, 0.3) !important;
        }
        
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
            color: #d1d5db !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
        }
        
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label[data-checked="true"] div[data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        /* Hide default Streamlit radio circles */
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label span[data-testid="stWidgetLabel"] {
            display: none !important;
        }
        
        /* Highlight sections */
        .highlight-box {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(14, 165, 233, 0.08));
            border-radius: 12px;
            padding: 16px;
            border: 1px solid rgba(14, 165, 233, 0.2);
            margin: 15px 0;
        }
        
        /* Adjust charts borders */
        .js-plotly-plot {
            border-radius: 12px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Load data helper with caching
@st.cache_data
def load_data():
    from utils.data_loader import load_data as get_data
    try:
        return get_data()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Main Entry Function
def main():
    # Inject styling
    inject_custom_styles()
    
    # Load dataset
    df = load_data()
    if df is None:
        return
        
    # Put dataset in session state so pages can access it
    st.session_state.df = df
    
    # Sidebar Header Design
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 15px 0 5px 0;">
            <span style="font-size: 2.6rem; filter: drop-shadow(0px 4px 10px rgba(14, 165, 233, 0.3));">⚖️</span>
            <h2 style="margin: 5px 0 0 0; font-family: 'Outfit', sans-serif; font-size: 1.45rem; background: linear-gradient(135deg, #10b981, #0ea5e9); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Gender Pay Gap Analysis</h2>
            <p style="color: #6b7280; font-size: 0.85rem; font-weight: 500; margin-top:2px;">Gender wage equity & bias sandbox</p>
        </div>
        <hr style="border-color: rgba(255,255,255,0.05); margin: 10px 0 20px 0;"/>
        """,
        unsafe_allow_html=True
    )
    
    # Custom Sidebar Navigation Menu
    st.sidebar.subheader("Navigation")
    pages = {
        "🏠  Problem & Summary": "Page1",
        "📊  Disparity Explorer": "Page2",
        "🔍  Root Cause Explorer": "Page3",
        "🤖  PayEquity AI & Simulator": "Page4"
    }
    
    # Select active page
    selected_page_name = st.sidebar.radio(
        "Go to page:",
        list(pages.keys()),
        label_visibility="collapsed"
    )
    
    active_page_module = pages[selected_page_name]
    
    # Sidebar footer
    st.sidebar.markdown(
        """
        <div style="font-size: 0.75rem; color: #4b5563; text-align: center; margin-top: 80px;">
            <p>Gender Pay Gap Analysis · 2026</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Dynamic routing
    if active_page_module == "Page1":
        import Page1
        Page1.show()
    elif active_page_module == "Page2":
        import Page2
        Page2.show()
    elif active_page_module == "Page3":
        import Page3
        Page3.show()
    elif active_page_module == "Page4":
        import Page4
        Page4.show()

if __name__ == "__main__":
    main()