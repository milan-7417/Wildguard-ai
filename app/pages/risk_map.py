import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import sys

# Append parent dir for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pages import prevention # import recommendation engine helpers

# Wayanad Study Area Bounding Box [lat_min, lon_min, lat_max, lon_max]
BBOX = [11.50, 75.80, 11.95, 76.40]

def haversine_dist(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2.0 * np.arcsin(np.sqrt(a)) * 6371.0

def render():
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <span class="badge badge-low" style="font-size: 0.85rem; padding: 6px 14px;">Spatial Analysis Console</span>
            <h1 style="font-size: 2.2rem; margin-top: 10px; margin-bottom: 5px;">Interactive HEC Risk Map</h1>
            <p style="color: #94a3b8; font-size: 1rem; margin-top: 0px;">
                Explore spatial risk scores, query environment covariates, and run local SHAP explainers by clicking on cells.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    df = st.session_state["data"]
    model = st.session_state["model"]
    explainer = st.session_state["explainer"]
    
    if df is None or model is None or explainer is None:
        st.warning("Please verify data preprocessing and training steps are complete.")
        return
        
    # Re-calculate risk scores if not present
    features = ["d_road", "d_water", "d_settlement", "d_forest", "forest_cover", "agricultural_exposure", "elevation", "slope"]
    if "risk_score" not in df.columns:
        probs = model.predict_proba(df[features])[:, 1]
        df["risk_prob"] = probs
        df["risk_score"] = (probs * 100).astype(int)
        
        def get_risk_level(score):
            if score < 25: return "Low"
            elif score < 50: return "Moderate"
            elif score < 75: return "High"
            else: return "Critical"
        df["risk_level"] = df["risk_score"].apply(get_risk_level)

    # 1. Filters Section
    st.markdown("<h4 style='margin-bottom:10px;'>Map Visualization Controls</h4>", unsafe_allow_html=True)
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
    
    with filter_col1:
        selected_levels = st.multiselect(
            "Filter Risk Levels:",
            options=["Low", "Moderate", "High", "Critical"],
            default=["Moderate", "High", "Critical"]
        )
        
    with filter_col2:
        map_style = st.selectbox(
            "Base Map Style:",
            options=["CartoDB Positron", "OpenStreetMap", "CartoDB DarkMatter"]
        )
        
    with filter_col3:
        # Search dropdown to quickly jump to key known hotspots
        hotspot_opts = {
            "Select from list to jump coordinates...": None,
            "Pulpally Fringe Hotspot (Cell ID 356)": 356,
            "Muthanga Sanctuary Border (Cell ID 198)": 198,
            "Thirunelli Migration Edge (Cell ID 655)": 655,
            "Vythiri Tea Estates (Cell ID 72)": 72,
            "Sulthan Bathery Interface (Cell ID 228)": 228
        }
        search_selection = st.selectbox("Search Hotspot Locations:", options=list(hotspot_opts.keys()))
        if hotspot_opts[search_selection] is not None:
            st.session_state["selected_cell_id"] = hotspot_opts[search_selection]

    # Filter dataframe
    filtered_df = df[df["risk_level"].isin(selected_levels)]
    
    # 2. Main Map Render and Click Layout
    map_display_col, detail_display_col = st.columns([5, 3])
    
    # Selected cell identifier
    sel_id = st.session_state["selected_cell_id"]
    # Fallback bounds check
    if sel_id >= len(df):
        sel_id = 0
        st.session_state["selected_cell_id"] = 0
        
    selected_row = df.iloc[sel_id]
    
    with map_display_col:
        # Initialize folium map
        # Center of Wayanad: 11.7258, 76.13
        m = folium.Map(
            location=[11.7258, 76.13], 
            zoom_start=10, 
            tiles=map_style.replace(" ", ""),
            control_scale=True
        )
        
        # Color codes with darker legibility contrast for light theme
        color_map = {
            "Low": "#059669",
            "Moderate": "#b45309",
            "High": "#c2410c",
            "Critical": "#dc2626"
        }
        
        # Draw filtered grid cells
        # To avoid lagging, we draw up to 400 cells. If more are visible, we warn the user.
        draw_limit = 450
        drawn_count = 0
        
        for idx, row in filtered_df.iterrows():
            if drawn_count >= draw_limit:
                break
                
            lvl = row["risk_level"]
            col = color_map[lvl]
            
            # Create rectangle coords
            bounds = [
                [row["lat_min"], row["lon_min"]],
                [row["lat_max"], row["lon_max"]]
            ]
            
            # Hover text
            tooltip_html = f"""
                <div style="font-family: sans-serif; font-size: 11px; padding: 4px;">
                    <b>Cell ID:</b> {int(row['cell_id'])}<br/>
                    <b>Risk Score:</b> <span style="color:{col}; font-weight:bold;">{int(row['risk_score'])}/100</span> ({lvl})<br/>
                    <b>Elevation:</b> {int(row['elevation'])}m
                </div>
            """
            
            # Highlight border if it is the active selected cell
            is_selected = (int(row["cell_id"]) == int(selected_row["cell_id"]))
            weight = 3 if is_selected else 0.5
            fill_opacity = 0.55 if is_selected else 0.35
            border_color = "#1b2c22" if is_selected else col
            
            rect = folium.Rectangle(
                bounds=bounds,
                color=border_color,
                weight=weight,
                fill=True,
                fill_color=col,
                fill_opacity=fill_opacity,
                tooltip=tooltip_html
            )
            rect.add_to(m)
            drawn_count += 1
            
        # Draw a highlighted pin at the selected cell centroid
        folium.CircleMarker(
            location=[selected_row["centroid_lat"], selected_row["centroid_lon"]],
            radius=6,
            color="#1b2c22",
            weight=2,
            fill=True,
            fill_color=color_map[selected_row["risk_level"]],
            fill_opacity=1.0,
            tooltip=f"Selected Location: Cell {int(selected_row['cell_id'])}"
        ).add_to(m)
        
        # Display the map in Streamlit
        st.markdown("<p style='font-size:0.8rem; color:#5c7265; margin-bottom:5px;'>💡 Click on any grid cell directly on the map, then wait 1s to sync data.</p>", unsafe_allow_html=True)
        
        # Call st_folium
        map_data = st_folium(
            m, 
            width="100%", 
            height=500,
            returned_objects=["last_clicked"]
        )
        
        # Update selected cell ID based on click
        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            
            # Check bounds to make sure the click lies inside study area
            if BBOX[0] <= click_lat <= BBOX[2] and BBOX[1] <= click_lon <= BBOX[3]:
                # Compute distance to all centroids
                distances = haversine_dist(
                    df["centroid_lat"].values, 
                    df["centroid_lon"].values, 
                    click_lat, 
                    click_lon
                )
                closest_idx = np.argmin(distances)
                # Avoid infinite refresh loops by only updating if selection changed
                if int(closest_idx) != int(st.session_state["selected_cell_id"]):
                    st.session_state["selected_cell_id"] = int(closest_idx)
                    st.rerun()

    # 3. Selected Location Details & Local SHAP
    with detail_display_col:
        st.markdown(f"<h3>Location Details: Cell {int(selected_row['cell_id'])}</h3>", unsafe_allow_html=True)
        
        lvl = selected_row["risk_level"]
        score = int(selected_row["risk_score"])
        col = color_map[lvl]
        
        # Display Score and Badge with premium light card styling
        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 1px solid #cbd5e1; padding: 20px; border-radius: 12px; margin-bottom: 20px; color: #0f172a; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span class="badge badge-{lvl.lower()}">{lvl} RISK</span>
                        <p style="margin: 5px 0 0 0; color: #334155; font-size: 0.92rem; font-weight: 600;">Coordinates: {selected_row['centroid_lat']:.4f}°N, {selected_row['centroid_lon']:.4f}°E</p>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 2.4rem; font-weight: 700; color: {col}; font-family: 'Space Grotesk', sans-serif;">{score}</span>
                        <span style="color: #334155; font-size: 1rem; font-weight: 600;">/100</span>
                    </div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Expandable environmental factors
        with st.expander("🌐 Environmental Attributes", expanded=True):
            st.markdown(
                f"""
                <table style="width: 100%; border-collapse: collapse; font-size: 0.92rem; color: #0f172a; line-height: 2.0;">
                    <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 6px 0; color:#334155; font-weight:600;">Forest Cover:</td><td style="text-align:right; font-weight:700; color:#0f172a;">{selected_row['forest_cover']:.1f}%</td></tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 6px 0; color:#334155; font-weight:600;">Agricultural Exposure:</td><td style="text-align:right; font-weight:700; color:#0f172a;">{selected_row['agricultural_exposure']:.1f}%</td></tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 6px 0; color:#334155; font-weight:600;">Distance to Settlement:</td><td style="text-align:right; font-weight:700; color:#0f172a;">{selected_row['d_settlement']:.2f} km</td></tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 6px 0; color:#334155; font-weight:600;">Distance to Forest Border:</td><td style="text-align:right; font-weight:700; color:#0f172a;">{selected_row['d_forest']:.2f} km</td></tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 6px 0; color:#334155; font-weight:600;">Distance to Waterway:</td><td style="text-align:right; font-weight:700; color:#0f172a;">{selected_row['d_water']:.2f} km</td></tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 6px 0; color:#334155; font-weight:600;">Distance to Major Road:</td><td style="text-align:right; font-weight:700; color:#0f172a;">{selected_row['d_road']:.2f} km</td></tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 6px 0; color:#334155; font-weight:600;">Elevation:</td><td style="text-align:right; font-weight:700; color:#0f172a;">{int(selected_row['elevation'])} m</td></tr>
                    <tr><td style="padding: 6px 0; color:#334155; font-weight:600;">Slope Angle:</td><td style="text-align:right; font-weight:700; color:#0f172a;">{selected_row['slope']:.1f}°</td></tr>
                </table>
                """, 
                unsafe_allow_html=True
            )
            
        # Calculate local SHAP values using cached TreeExplainer
        # Features map names
        feature_names_clean = {
            "d_road": "Road Proximity",
            "d_water": "Water Proximity",
            "d_settlement": "Settlement Proximity",
            "d_forest": "Forest Proximity",
            "forest_cover": "Forest Cover",
            "agricultural_exposure": "Agricultural Exposure",
            "elevation": "Elevation",
            "slope": "Slope"
        }
        
        # Calculate on the fly for the selected cell
        # We index the global shap values
        X = df[features]
        # Generate SHAP explanation for the single selected row
        row_features = X.iloc[[sel_id]]
        shap_values_row = explainer(row_features)
        
        # Extract base value and local values
        # shap_values_row is a Explanation object. Values are in log-odds.
        local_shap = shap_values_row.values[0]
        base_value = shap_values_row.base_values[0]
        
        # Construct dataframe of contributions
        contributions = []
        for feat, val in zip(features, local_shap):
            contributions.append({
                "Feature": feature_names_clean[feat],
                "Contribution": float(val)
            })
        cont_df = pd.DataFrame(contributions)
        
        # Separate positive (increasing risk) and negative (decreasing risk) drivers
        pos_drivers = cont_df[cont_df["Contribution"] > 0].sort_values("Contribution", ascending=False)
        neg_drivers = cont_df[cont_df["Contribution"] <= 0].sort_values("Contribution", ascending=True)
        
        # Display Local SHAP Drivers
        with st.expander("🔍 Explainable AI: Risk Drivers", expanded=True):
            st.markdown("<p style='font-size: 0.92rem; color:#334155; margin-top:0; font-weight: 500;'>Top contributing factors to this cell's risk score (SHAP log-odds offsets):</p>", unsafe_allow_html=True)
            
            # Draw positive drivers
            st.markdown("<b style='color:#dc2626; font-size:0.95rem; font-weight:700;'>Increasing Risk:</b>", unsafe_allow_html=True)
            for idx, r in pos_drivers.head(3).iterrows():
                val_str = f"+{r['Contribution']:.2f}"
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; font-size:0.92rem; padding: 3px 0; font-weight:600; color:#0f172a;">
                        <span>{r['Feature']}</span><span style="color:#dc2626; font-weight:700;">{val_str}</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
            # Draw negative drivers
            st.markdown("<br><b style='color:#047857; font-size:0.95rem; font-weight:700;'>Decreasing Risk:</b>", unsafe_allow_html=True)
            for idx, r in neg_drivers.head(3).iterrows():
                val_str = f"{r['Contribution']:.2f}"
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; font-size:0.92rem; padding: 3px 0; font-weight:600; color:#0f172a;">
                        <span>{r['Feature']}</span><span style="color:#047857; font-weight:700;">{val_str}</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
        # Targeted recommendations summary
        recs = prevention.get_recommendations(selected_row)
        with st.expander("📋 Action Recommendations", expanded=True):
            st.markdown(f"<p style='margin:0; font-size:0.95rem; font-weight:700; color:{col};'>Priority Level: {lvl.upper()}</p>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <ol style="font-size: 0.95rem; padding-left: 18px; margin: 8px 0 0 0; color: #0f172a; line-height: 1.6; font-weight: 500;">
                    {"".join([f"<li style='margin-bottom:6px;'>{act['action']}</li>" for act in recs[:3]])}
                </ol>
                """,
                unsafe_allow_html=True
            )
