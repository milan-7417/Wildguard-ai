import streamlit as st
import pandas as pd
import numpy as np
import os

def get_recommendations(row):
    """
    Core Recommendation Engine. Evaluates cell-level spatial metrics
    and returns tailored, prioritized prevention strategies.
    """
    recs = []
    
    # 1. Settlement proximity
    if row["d_settlement"] < 2.5:
        recs.append({
            "action": "Acoustic Sensors & SMS Alert Warning Networks",
            "desc": "Deploy automated acoustic tripwire sensors that detect elephant vocalizations (rumbles) and trigger instant SMS coordinate broadcasts to adjacent villages.",
            "category": "Early Warning",
            "priority": "Critical"
        })
        recs.append({
            "action": "Primary Response Team (PRT) Mobilization",
            "desc": "Train and equip local village volunteers with high-intensity flashlights, sirens, and communication systems to safely coordinate herd deterrent activities.",
            "category": "Community Support",
            "priority": "High"
        })
        
    # 2. Forest fragmentation / Edge cover
    if 15.0 <= row["forest_cover"] <= 65.0:
        recs.append({
            "action": "Buffer-Zone Alternative Crop Cultivation",
            "desc": "Encourage community farmers to transition crop margins from highly palatable plants (banana, sugarcane, paddy) to elephant-repellent alternatives like chili, ginger, lemon, and bee-hive fences.",
            "category": "Agriculture",
            "priority": "High"
        })
        recs.append({
            "action": "Eco-Restoration of Edge Fragments",
            "desc": "Prioritize micro-corridor reforestation inside fragmented loops. Swapping invasive weeds like Lantana camara with native fodder grass reduces elephant foraging on farm borders.",
            "category": "Habitat",
            "priority": "Moderate"
        })
        
    # 3. High agricultural exposure
    if row["agricultural_exposure"] > 40.0:
        recs.append({
            "action": "Smart Electric Fencing & Trenches",
            "desc": "Construct solar-powered hanging electric fences along the forest-farm interface. Maintain wide trenches (Elephant Proof Trenches - EPTs) and schedule regular community weed clearing to prevent short circuits.",
            "category": "Barrier Protection",
            "priority": "Critical"
        })
        recs.append({
            "action": "Community Grain Bank Safes",
            "desc": "Build reinforced, central grain storage vaults in villages. Secure storage prevents elephants from smelling stored crops and raiding domestic houses.",
            "category": "Infrastructure",
            "priority": "Moderate"
        })
        
    # 4. Road Proximity
    if row["d_road"] < 1.5:
        recs.append({
            "action": "Highway Crossing Traffic Management",
            "desc": "Implement seasonal night-traffic restrictions (e.g. 6 PM to 6 AM) on state highways and NH 766 cutting through Muthanga and Begur to give elephants safe crossing corridors.",
            "category": "Infrastructure",
            "priority": "High"
        })
        recs.append({
            "action": "Smart Wildlife Crossing Signs & Speed Limiters",
            "desc": "Install thermal cameras linked to digital signage to alert oncoming vehicle drivers when elephants approach the road, coupled with speed-calming humps.",
            "category": "Infrastructure",
            "priority": "Moderate"
        })
        
    # 5. Water Availability
    if row["d_water"] < 2.0:
        recs.append({
            "action": "Habitat Water Resource Recharge",
            "desc": "Protect natural forest waterholes from weed choking. Install solar-powered borewells in core forests to recharge ponds during the dry summer, discouraging elephants from migrating into agricultural irrigation canals.",
            "category": "Habitat",
            "priority": "High"
        })
        
    # Fallback
    if not recs:
        recs.append({
            "action": "Baseline Wildlife Tracking & Forest Patrols",
            "desc": "Maintain routine anti-poaching foot patrols and monitor vegetation changes. Establish camera traps to audit native wildlife presence.",
            "category": "Monitoring",
            "priority": "Low"
        })
        
    return recs

def render():
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <span class="badge badge-low" style="font-size: 0.85rem; padding: 6px 14px;">Mitigation Command Center</span>
            <h1 style="font-size: 2.2rem; margin-top: 10px; margin-bottom: 5px;">Action Prevention Center</h1>
            <p style="color: #334155; font-size: 1.05rem; margin-top: 0px; font-weight: 500;">
                Review target recommendations and plan conservation intervention campaigns.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Disclaimer
    st.info(
        "🛡️ **Decision-Support Wording**: The strategies listed below are decision-support recommendations generated "
        "by the environmental vulnerability indicators of the selected zone. They represent scientifically supported "
        "conflict prevention guidelines, not guaranteed solutions."
    )
    
    df = st.session_state["data"]
    if df is None:
        st.warning("Please run data preprocessing first.")
        return
        
    # Load selected cell
    sel_id = st.session_state["selected_cell_id"]
    if sel_id >= len(df):
        sel_id = 0
    selected_row = df.iloc[sel_id]
    
    # Layout splits
    header_col, select_col = st.columns([2, 1])
    with header_col:
        st.markdown(f"<h3>Active Vulnerability Target: Cell {int(selected_row['cell_id'])}</h3>", unsafe_allow_html=True)
    with select_col:
        # Manual cell ID input search box
        manual_id = st.number_input(
            "Select Cell ID manually:", 
            min_value=0, 
            max_value=len(df)-1, 
            value=int(selected_row["cell_id"]),
            step=1
        )
        if manual_id != int(selected_row["cell_id"]):
            st.session_state["selected_cell_id"] = int(manual_id)
            st.rerun()

    # Get local details
    recs = get_recommendations(selected_row)
    
    # Priority color map for styling badges
    badge_colors = {
        "Critical": "crit",
        "High": "high",
        "Moderate": "mod",
        "Low": "low"
    }
    
    # Render recommendations as rich UI Cards
    st.markdown("<h4 style='color: #047857; text-transform: uppercase; margin-bottom: 20px; font-weight:700;'>Tailored Mitigation Measures</h4>", unsafe_allow_html=True)
    
    for r in recs:
        badge_style = badge_colors.get(r["priority"], "low")
        st.markdown(
            f"""
            <div class="premium-card" style="border-color: #cbd5e1; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
                    <div>
                        <span class="badge badge-{r['category'].lower().replace(' ', '')}" style="background-color:rgba(5, 150, 105, 0.14); color:#047857; border:1px solid rgba(5, 150, 105, 0.3); font-size:0.75rem; padding: 3px 10px;">{r['category']}</span>
                        <h4 style="margin: 8px 0 0 0; font-size: 1.25rem; color:#0f172a; font-weight: 700;">{r['action']}</h4>
                    </div>
                    <span class="badge badge-{badge_style}">{r['priority']} Priority</span>
                </div>
                <p style="margin: 8px 0 0 0; font-size: 0.95rem; color:#334155; line-height:1.6; font-weight: 500;">{r['desc']}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
