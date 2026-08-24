import os
import streamlit as st

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "final_features_with_clusters.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "xgb_model.json")
EXPLAINER_FILE = os.path.join(BASE_DIR, "models", "shap_explainer.pkl")
METRICS_FILE = os.path.join(BASE_DIR, "models", "metrics.json")

# Styling Colors
COLOR_PALETTE = {
    "background": "#f8fafc",        # Premium clean white/slate
    "card_bg": "#ffffff",           # Pure white card
    "card_hover": "#fafcfb",        # Soft card hover highlight
    "accent": "#059669",            # Rich forest green accent
    "text_primary": "#0f172a",      # Deep slate for highest legibility
    "text_secondary": "#334155",    # Slate 700 for highly readable small text
    "border": "#e2e8f0",            # Clean slate border
    "risk_low": "#047857",          # Darker green for high contrast on light backgrounds
    "risk_mod": "#b45309",          # Darker amber
    "risk_high": "#c2410c",         # Darker orange
    "risk_crit": "#dc2626"          # Red
}

def apply_theme():
    """Injects premium custom CSS to standard Streamlit layouts to match a clean light sage conservation console."""
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600&display=swap');
        
        /* Apply fonts and background */
        html, body, [class*="css"], .stApp {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #f8fafc !important;
            color: #0f172a !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700 !important;
            color: #0f172a !important;
            letter-spacing: -0.025em;
        }
        
        /* Metric cards styling */
        div[data-testid="stMetric"] {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            padding: 20px 24px !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        div[data-testid="stMetric"]:hover {
            border-color: #059669 !important;
            box-shadow: 0 10px 15px -3px rgba(5, 150, 105, 0.08), 0 4px 6px -2px rgba(5, 150, 105, 0.03) !important;
            transform: translateY(-2px);
        }
        div[data-testid="stMetric"] label {
            color: #334155 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 700 !important;
            font-size: 2.2rem !important;
        }
        
        /* Custom Premium Card class */
        .premium-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 26px;
            border-radius: 14px;
            margin-bottom: 22px;
            color: #1e293b;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .premium-card:hover {
            border-color: #059669;
            box-shadow: 0 12px 20px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
            transform: translateY(-2px);
        }
        
        /* Custom badge */
        .badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .badge-low { background-color: rgba(5, 150, 105, 0.12); color: #047857; border: 1px solid rgba(5, 150, 105, 0.25); }
        .badge-mod { background-color: rgba(180, 83, 9, 0.12); color: #92400e; border: 1px solid rgba(180, 83, 9, 0.25); }
        .badge-high { background-color: rgba(194, 65, 12, 0.12); color: #9a3412; border: 1px solid rgba(194, 65, 12, 0.25); }
        .badge-crit { background-color: rgba(220, 38, 38, 0.12); color: #991b1b; border: 1px solid rgba(220, 38, 38, 0.25); }

        /* Style Streamlit buttons */
        .stButton>button {
            background-color: #059669 !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border: none !important;
            padding: 12px 26px !important;
            border-radius: 8px !important;
            font-size: 0.95rem !important;
            box-shadow: 0 2px 4px rgba(5, 150, 105, 0.15) !important;
            transition: all 0.2s ease !important;
        }
        .stButton>button:hover {
            background-color: #047857 !important;
            box-shadow: 0 6px 15px rgba(4, 120, 87, 0.3) !important;
            transform: scale(1.01);
        }
        
        /* Streamlit Sidebar (light grey tint with high contrast) */
        section[data-testid="stSidebar"] {
            background-color: #f1f5f9 !important;
            border-right: 1px solid #e2e8f0 !important;
        }
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
            color: #0f172a !important;
        }
        
        /* Selectbox & Inputs styling */
        div[data-baseweb="select"] {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
        }
        div[data-baseweb="select"] * {
            color: #0f172a !important;
        }
        
        /* Hide default Streamlit sidebar page navigation list */
        div[data-testid="stSidebarNav"] {
            display: none !important;
        }
        
        /* Footer text styling */
        .footer-text {
            color: #475569;
            font-size: 0.92rem;
            text-align: center;
            margin-top: 45px;
            border-top: 1px solid #e2e8f0;
            padding-top: 25px;
        }
        </style>
    """, unsafe_allow_html=True)
