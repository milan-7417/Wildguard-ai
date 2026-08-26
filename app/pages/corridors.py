import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

def calculate_corridor_score(row):
    """
    Computes a transparent, ecologically based corridor suitability score (0-100).
    A high score represents a high-suitability corridor (high forest, low slope, far from settlements/roads).
    """
    # 1. Forest Connectivity (0-100): Direct mapping of forest cover
    forest_conn = row["forest_cover"]
    
    # 2. Water Availability (0-100): High score when water is close
    water_avail = 100.0 * np.exp(-row["d_water"] / 3.0)
    
    # 3. Lack of Human Pressure (0-100): High score when settlements are far away
    # e.g., d_settlement > 5km gets near 100.
    human_pressure = 100.0 * (1.0 - np.exp(-row["d_settlement"] / 4.0))
    
    # 4. Lack of Road Barriers (0-100): High score when roads are far away
    road_barriers = 100.0 * (1.0 - np.exp(-row["d_road"] / 2.5))
    
    # 5. Habitat Quality (0-100): Preferred low-slope, high-cover valleys
    slope_factor = max(0, 100.0 - (row["slope"] * 2.2)) # drops to 0 at ~45 deg
    habitat_quality = 0.6 * row["forest_cover"] + 0.4 * slope_factor
    
    # Weighted average corridor suitability
    suitability = (
        0.3 * forest_conn + 
        0.2 * water_avail + 
        0.2 * human_pressure + 
        0.1 * road_barriers + 
        0.2 * habitat_quality
    )
    
    return {
        "suitability": int(suitability),
        "forest_conn": int(forest_conn),
        "water_avail": int(water_avail),
        "human_pressure": int(human_pressure),
        "road_barriers": int(road_barriers),
        "habitat_quality": int(habitat_quality)
    }

def render():
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <span class="badge badge-low" style="font-size: 0.85rem; padding: 6px 14px;">Landscape Connectivity Console</span>
            <h1 style="font-size: 2.2rem; margin-top: 10px; margin-bottom: 5px;">Habitat & Corridor Analysis</h1>
            <p style="color: #94a3b8; font-size: 1rem; margin-top: 0px;">
                Identify potential wildlife connectivity pathways and evaluate habitat quality parameters.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Disclaimer
    st.info(
        "🛡️ **Scientific Disclaimer**: These calculations represent potential habitat connectivity zones based "
        "on ecological factors. They are not officially designated corridors and should be used strictly "
        "as a decision-support guide for wildlife crossing management."
    )
    
    df = st.session_state["data"]
    if df is None:
        st.warning("Please run data preprocessing first.")
        return
        
    # Calculate scores for all rows
    if "corridor_suitability" not in df.columns:
        suit_scores = []
        for idx, row in df.iterrows():
            res = calculate_corridor_score(row)
            suit_scores.append(res)
            
        df["corridor_suitability"] = [r["suitability"] for r in suit_scores]
        df["forest_conn"] = [r["forest_conn"] for r in suit_scores]
        df["water_avail"] = [r["water_avail"] for r in suit_scores]
        df["human_pressure"] = [r["human_pressure"] for r in suit_scores]
        df["road_barriers"] = [r["road_barriers"] for r in suit_scores]
        df["habitat_quality"] = [r["habitat_quality"] for r in suit_scores]
        
    # Main split layout
    map_col, local_col = st.columns([5, 3])
    
    # Get selected cell details
    sel_id = st.session_state["selected_cell_id"]
    if sel_id >= len(df):
        sel_id = 0
    selected_row = df.iloc[sel_id]
    
    with map_col:
        st.markdown("<h3 style='margin-bottom:15px;'>Potential Habitat Connectivity Zones</h3>", unsafe_allow_html=True)
        
        # Plotly density map of corridor suitability (light style) (handling Plotly v6 map/mapbox deprecations)
        if hasattr(px, "density_map"):
            fig = px.density_map(
                df, 
                lat="centroid_lat", 
                lon="centroid_lon", 
                z="corridor_suitability", 
                radius=16,
                center=dict(lat=11.7258, lon=76.10), 
                zoom=9.5,
                map_style="carto-positron",
                color_continuous_scale="Viridis", # Beautiful purple-to-yellow scale
                labels={"corridor_suitability": "Suitability"},
                title=None
            )
        else:
            fig = px.density_mapbox(
                df, 
                lat="centroid_lat", 
                lon="centroid_lon", 
                z="corridor_suitability", 
                radius=16,
                center=dict(lat=11.7258, lon=76.10), 
                zoom=9.5,
                mapbox_style="carto-positron",
                color_continuous_scale="Viridis", # Beautiful purple-to-yellow scale
                labels={"corridor_suitability": "Suitability"},
                title=None
            )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar=dict(
                title="Suitability Score",
                thickness=15,
                len=0.7,
                x=0.98,
                title_font_color="#1b2c22",
                tickfont_color="#1b2c22"
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown(
            "<p style='color: #334155; font-size: 0.92rem; text-align: center; margin-top: 8px; font-weight: 500;'>"
            "Note: High scores (yellow/green) indicate potential natural pathways that elephants are likely to traverse due to forest density and low human disturbance."
            "</p>", 
            unsafe_allow_html=True
        )
        
    with local_col:
        st.markdown(f"<h3>Cell {int(selected_row['cell_id'])} Connectivity Profile</h3>", unsafe_allow_html=True)
        
        score = int(selected_row["corridor_suitability"])
        # Determine color of score
        if score >= 70: score_color = "#047857" # green
        elif score >= 40: score_color = "#92400e" # yellow/amber
        else: score_color = "#475569" # grey/slate
        
        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 1px solid #cbd5e1; padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: center; color: #0f172a; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);">
                <p style="margin: 0; color: #334155; font-size: 0.95rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Corridor Suitability</p>
                <div style="margin-top: 8px;">
                    <span style="font-size: 3.5rem; font-weight: 700; color: {score_color}; font-family: 'Space Grotesk', sans-serif;">{score}</span>
                    <span style="color: #334155; font-size: 1.15rem; font-weight: 600;">/100</span>
                </div>
                <p style="margin: 8px 0 0 0; color: #0f172a; font-size: 1.05rem; font-weight: 700;">
                    { "High Habitat Connectivity" if score >= 70 else "Moderate Connectivity Zone" if score >= 40 else "High Human Pressure Zone" }
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Display Breakdown as progress bars
        st.markdown("<h4 style='font-size: 1rem; margin-bottom: 15px; color: #059669; text-transform: uppercase; font-weight:700;'>Suitability Indicators</h4>", unsafe_allow_html=True)
        
        metrics_dict = {
            "Forest Connectivity": selected_row["forest_conn"],
            "Water Resource Proximity": selected_row["water_avail"],
            "Lack of Settlement Intrusion": selected_row["human_pressure"],
            "Road Passage Openness": selected_row["road_barriers"],
            "Habitat Quality Core": selected_row["habitat_quality"]
        }
        
        for name, val in metrics_dict.items():
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; font-size: 0.95rem; margin-bottom: 3px;">
                    <span style="color: #334155; font-weight:600;">{name}</span>
                    <span style="font-weight:700; color: #0f172a;">{val}/100</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.progress(float(val) / 100.0)
            st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)
            
        # Explanatory card
        st.markdown(
            """
            <div class="premium-card" style="padding: 15px; margin-top: 15px; margin-bottom: 0px; font-size: 0.95rem; line-height: 1.6; color: #334155; border-color: #cbd5e1; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);">
                <b>Conservation Tip:</b> High-suitability connectivity cells should be preserved against new commercial fence permits, highway speed expansion, or settlement layouts to prevent blocking migration lanes.
            </div>
            """, 
            unsafe_allow_html=True
        )
