import os
import pickle
import pandas as pd
import streamlit as st
import xgboost as xgb
import json

# Import config and page renderers
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# Page imports
from pages import overview, risk_map, risk_analysis, corridors, prevention

# Initialize page config
st.set_page_config(
    page_title="WildGuard AI | Human-Elephant Conflict Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources (with caching to prevent reloading on page switch)
@st.cache_resource
def load_dataset():
    if os.path.exists(config.DATA_FILE):
        return pd.read_csv(config.DATA_FILE)
    else:
        st.error(f"Dataset not found at {config.DATA_FILE}. Please run preprocessing first.")
        return None

@st.cache_resource
def load_model():
    if os.path.exists(config.MODEL_FILE):
        model = xgb.XGBClassifier()
        model.load_model(config.MODEL_FILE)
        return model
    else:
        st.error(f"Model file not found at {config.MODEL_FILE}. Please run training script first.")
        return None

@st.cache_resource
def load_explainer():
    if os.path.exists(config.EXPLAINER_FILE):
        with open(config.EXPLAINER_FILE, "rb") as f:
            return pickle.load(f)
    else:
        st.error(f"SHAP Explainer file not found at {config.EXPLAINER_FILE}. Please run training script first.")
        return None

@st.cache_resource
def load_metrics():
    if os.path.exists(config.METRICS_FILE):
        with open(config.METRICS_FILE, "r") as f:
            return json.load(f)
    else:
        st.error(f"Metrics file not found at {config.METRICS_FILE}. Please run training script first.")
        return None

def main():
    # Apply custom styling theme
    config.apply_theme()
    
    # Load data into session state
    if "data" not in st.session_state:
        st.session_state["data"] = load_dataset()
    if "model" not in st.session_state:
        st.session_state["model"] = load_model()
    if "explainer" not in st.session_state:
        st.session_state["explainer"] = load_explainer()
    if "metrics" not in st.session_state:
        st.session_state["metrics"] = load_metrics()
        
    # Default selection state for map clicking
    if "selected_cell_id" not in st.session_state:
        # Default to a high-risk cell in Pulpally hotspot
        st.session_state["selected_cell_id"] = 356 # fallback index
        
    # Sidebar Branding
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px 0px;">
            <h2 style="color: #059669; margin-bottom: 0px; font-weight: 700; font-size: 1.6rem;">🛡️ WILDGUARD AI</h2>
            <p style="color: #334155; font-size: 0.92rem; margin-top: 5px; font-weight: 600;">
                Human-Wildlife Conflict Mitigation
            </p>
        </div>
        <hr style="border-color: #cbd5e1; margin-top: 5px; margin-bottom: 20px;"/>
        """, 
        unsafe_allow_html=True
    )
    
    # Sidebar Navigation Router
    st.sidebar.markdown("<h3 style='font-size: 1.05rem; color: #059669; text-transform: uppercase; letter-spacing: 0.05em; font-weight:700;'>Navigation</h3>", unsafe_allow_html=True)
    page = st.sidebar.radio(
        label="Select Dashboard Module",
        options=[
            "1. Overview Console",
            "2. Interactive Risk Map",
            "3. Explainable ML Analysis",
            "4. Habitat & Corridors",
            "5. Action Prevention Center"
        ],
        label_visibility="collapsed"
    )
    
    # Study Area Information
    st.sidebar.markdown("<br><hr style='border-color: #cbd5e1; margin-bottom: 15px;'/><h3 style='font-size: 1.05rem; color: #059669; text-transform: uppercase; letter-spacing: 0.05em; font-weight:700;'>Active Landscape</h3>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="background-color: #ffffff; border: 1px solid #cbd5e1; padding: 16px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <p style="margin: 0; font-weight: 700; color: #0f172a; font-size: 1.05rem;">Wayanad District</p>
            <p style="margin: 2px 0 10px 0; color: #334155; font-size: 0.88rem; font-weight: 500;">Kerala, India (Western Ghats)</p>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; color: #334155; line-height: 1.6;">
                <tr><td style="padding: 3px 0; font-weight: 600;">Grid Scale:</td><td style="text-align: right; color: #0f172a; font-weight: 700;">0.02° (~2.2km)</td></tr>
                <tr><td style="padding: 3px 0; font-weight: 600;">Cell Count:</td><td style="text-align: right; color: #0f172a; font-weight: 700;">713</td></tr>
                <tr><td style="padding: 3px 0; font-weight: 600;">Species Focus:</td><td style="text-align: right; color: #059669; font-weight: 700;">Asian Elephant</td></tr>
                <tr><td style="padding: 3px 0; font-weight: 600;">Model Version:</td><td style="text-align: right; color: #0f172a; font-weight: 700;">XGBoost v1.0</td></tr>
            </table>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # System Status indicators
    st.sidebar.markdown("<br><h3 style='font-size: 0.95rem; color: #334155; text-transform: uppercase; font-weight:700;'>Console Status</h3>", unsafe_allow_html=True)
    status_color = "#059669" if st.session_state["model"] is not None else "#ef4444"
    status_text = "Operational" if st.session_state["model"] is not None else "Model Missing"
    st.sidebar.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.88rem; color: #334155; font-weight: 600;">
            <span style="height: 10px; width: 10px; background-color: {status_color}; border-radius: 50%; display: inline-block;"></span>
            <span>AI Risk Core: <b style="color: #0f172a;">{status_text}</b></span>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Route Pages
    if page.startswith("1."):
        overview.render()
    elif page.startswith("2."):
        risk_map.render()
    elif page.startswith("3."):
        risk_analysis.render()
    elif page.startswith("4."):
        corridors.render()
    elif page.startswith("5."):
        prevention.render()
        
    # Footer
    st.markdown(
        """
        <div class="footer-text">
            <b>WildGuard AI</b> • AI-Powered Human–Wildlife Conflict Prediction & Prevention • Hack the Habitat 2026<br/>
            <i>"Estimating spatial human–elephant conflict risk using historical patterns and environmental covariates."</i>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
