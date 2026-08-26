import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

def render():
    st.markdown(
        """
        <div style="margin-bottom: 25px;">
            <span class="badge badge-low" style="font-size: 0.85rem; padding: 6px 14px;">Landscape Decision Support Console</span>
            <h1 style="font-size: 2.8rem; margin-top: 10px; margin-bottom: 5px;">WILDGUARD AI</h1>
            <p style="color: #94a3b8; font-size: 1.2rem; margin-top: 0px;">
                AI-Powered Human–Wildlife Conflict Prediction & Prevention
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Load dataset
    df = st.session_state["data"]
    model = st.session_state["model"]
    
    if df is None or model is None:
        st.warning("Awaiting datasets and trained models. Please complete the preprocessing and training phases.")
        return
        
    # Make model predictions on all grid cells
    features = ["d_road", "d_water", "d_settlement", "d_forest", "forest_cover", "agricultural_exposure", "elevation", "slope"]
    X = df[features]
    
    # Compute conflict risk probability using the trained XGBoost model
    probs = model.predict_proba(X)[:, 1]
    df["risk_prob"] = probs
    df["risk_score"] = (probs * 100).astype(int)
    
    # Define Risk Categories
    # Low: < 25%, Moderate: 25-50%, High: 50-75%, Critical: >= 75%
    def get_risk_level(score):
        if score < 25: return "Low"
        elif score < 50: return "Moderate"
        elif score < 75: return "High"
        else: return "Critical"
        
    df["risk_level"] = df["risk_score"].apply(get_risk_level)
    
    # Calculate statistics for KPIs
    high_crit_df = df[df["risk_level"].isin(["High", "Critical"])]
    avg_risk = df["risk_score"].mean()
    high_risk_zones = len(df[df["risk_level"] == "High"])
    crit_risk_zones = len(df[df["risk_level"] == "Critical"])
    
    # Grid cell size = 0.02° latitude x 0.02° longitude ~ 2.22 km x 2.22 km = 4.93 km² per cell
    cell_area_km2 = 4.93
    total_area_at_risk = len(high_crit_df) * cell_area_km2
    
    # Render KPI Cards in a row
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.metric(
            label="⚠️ High-Risk Grid Zones",
            value=f"{high_risk_zones} cells",
            delta=f"Area: {high_risk_zones * cell_area_km2:.1f} km²",
            delta_color="off"
        )
        
    with kpi_col2:
        st.metric(
            label="🔴 Critical-Risk Zones",
            value=f"{crit_risk_zones} cells",
            delta=f"Immediate priority",
            delta_color="inverse"
        )
        
    with kpi_col3:
        st.metric(
            label="📊 Average Risk Score",
            value=f"{avg_risk:.1f} / 100",
            delta="Landscape average",
            delta_color="off"
        )
        
    with kpi_col4:
        st.metric(
            label="🌳 Habitat Area At Risk",
            value=f"{total_area_at_risk:.1f} km²",
            delta=f"{len(high_crit_df)/len(df)*100:.1f}% of study area",
            delta_color="off"
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main content split layout
    map_col, info_col = st.columns([2, 1])
    
    with map_col:
        st.markdown("<h3 style='margin-bottom:15px;'>Landscape Conflict Risk Heatmap</h3>", unsafe_allow_html=True)
        # Create a beautiful Plotly density map with light theme style (handling Plotly v6 map/mapbox deprecations)
        if hasattr(px, "density_map"):
            fig = px.density_map(
                df, 
                lat="centroid_lat", 
                lon="centroid_lon", 
                z="risk_score", 
                radius=15,
                center=dict(lat=11.7258, lon=76.10), 
                zoom=9.5,
                map_style="carto-positron",
                color_continuous_scale=[
                    [0.0, "rgba(5, 150, 105, 0.1)"],     # Low (transparent green)
                    [0.25, "rgba(5, 150, 105, 0.4)"],
                    [0.5, "rgba(180, 83, 9, 0.7)"],      # Mod (amber)
                    [0.75, "rgba(194, 65, 12, 0.85)"],    # High (orange)
                    [1.0, "rgba(220, 38, 38, 1.0)"]       # Critical (red)
                ],
                labels={"risk_score": "Risk Score"},
                title=None
            )
        else:
            fig = px.density_mapbox(
                df, 
                lat="centroid_lat", 
                lon="centroid_lon", 
                z="risk_score", 
                radius=15,
                center=dict(lat=11.7258, lon=76.10), 
                zoom=9.5,
                mapbox_style="carto-positron",
                color_continuous_scale=[
                    [0.0, "rgba(5, 150, 105, 0.1)"],     # Low (transparent green)
                    [0.25, "rgba(5, 150, 105, 0.4)"],
                    [0.5, "rgba(180, 83, 9, 0.7)"],      # Mod (amber)
                    [0.75, "rgba(194, 65, 12, 0.85)"],    # High (orange)
                    [1.0, "rgba(220, 38, 38, 1.0)"]       # Critical (red)
                ],
                labels={"risk_score": "Risk Score"},
                title=None
            )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=True,
            coloraxis_colorbar=dict(
                title="Risk Score",
                thickness=15,
                len=0.7,
                x=0.98,
                title_font_color="#0f172a",
                tickfont_color="#0f172a"
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown(
            "<p style='color: #334155; font-size: 0.92rem; text-align: center; margin-top: 8px; font-weight: 500;'> "
            "Note: Heatmap represents model-predicted spatial risk scores (0-100) based on local environmental factors."
            "</p>", 
            unsafe_allow_html=True
        )
        
    with info_col:
        st.markdown("<h3 style='margin-bottom:15px;'>Executive Action Plan</h3>", unsafe_allow_html=True)
        
        # Calculate feature importances from final model to show top risk factors
        importance = model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': [
                "Road Proximity", "Water Proximity", "Settlement Proximity", 
                "Forest Proximity", "Forest Cover Density", "Agricultural Exposure", 
                "Elevation", "Slope"
            ],
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Render top risk factors as a beautiful horizontal bar chart
        st.markdown(
            "<div style='background-color: #ffffff; border: 1px solid #cbd5e1; padding: 18px; border-radius: 10px; margin-bottom: 20px; color: #0f172a;'> "
            "<h4 style='margin-top:0; margin-bottom: 12px; color: #059669; font-size: 1rem; text-transform: uppercase; font-weight: 700;'>Key Regional Risk Drivers</h4>",
            unsafe_allow_html=True
        )
        
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            y=feature_imp_df['Feature'][::-1],
            x=feature_imp_df['Importance'][::-1],
            orientation='h',
            marker=dict(
                color='#059669',
                line=dict(color='#059669', width=1)
            )
        ))
        fig_imp.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=180,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor='#e2e8f0', tickfont=dict(color='#334155', size=10)),
            yaxis=dict(tickfont=dict(color='#0f172a', size=10))
        )
        st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Priority recommendations summary card
        st.markdown(
            """
            <div class="premium-card" style="padding: 20px; margin-bottom: 0px; border-color: #cbd5e1; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
                <h4 style="margin-top:0; color: #dc2626; font-size: 1.05rem; text-transform: uppercase; font-weight: 700;">
                    📢 Critical Alert Recommendations
                </h4>
                <p style="font-size: 0.92rem; color: #334155; margin-top: 5px; margin-bottom: 15px; font-weight: 500;">
                    Grid locations with predicted risk scores exceeding 75 require active mitigation:
                </p>
                <ol style="font-size: 0.92rem; padding-left: 20px; margin: 0; color: #0f172a; line-height: 1.7; font-weight: 500;">
                    <li style="margin-bottom: 10px;">
                        <b>Panamaram & Pulpally borders</b>: Set up primary community response teams (PRTs) and deploy SMS-based early warning coordinates.
                    </li>
                    <li style="margin-bottom: 10px;">
                        <b>Muthanga Forest edge</b>: Restrict road operations on NH 766 during peak conflict hours (6:00 PM to 6:00 AM).
                    </li>
                    <li style="margin-bottom: 0px;">
                        <b>Wayanad Sanctuary periphery</b>: Inspect and reinforce solar electric fencing grids near agricultural margins.
                    </li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )
