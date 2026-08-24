import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve
import os

def render():
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <span class="badge badge-low" style="font-size: 0.85rem; padding: 6px 14px;">Scientific Validation Console</span>
            <h1 style="font-size: 2.2rem; margin-top: 10px; margin-bottom: 5px;">Explainable ML & Performance</h1>
            <p style="color: #94a3b8; font-size: 1rem; margin-top: 0px;">
                Inspect spatial cross-validation metrics, compare models, and understand global environmental drivers.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Disclaimer
    st.info(
        "🛡️ **Scientific Disclaimer**: WildGuard AI estimates spatial human–elephant conflict risk "
        "using historical conflict patterns and environmental/geospatial factors. It does NOT "
        "predict exact elephant movement paths or track individual animals in real-time."
    )
    
    df = st.session_state["data"]
    metrics = st.session_state["metrics"]
    explainer = st.session_state["explainer"]
    
    if df is None or metrics is None or explainer is None:
        st.warning("Please ensure training metrics and SHAP explainers have been saved.")
        return
        
    # Tab Layout
    tab_perf, tab_shap = st.tabs(["📊 Model Performance Comparisons", "🌐 Global Environmental SHAP Drivers"])
    
    with tab_perf:
        st.markdown("<h3>Spatial Cross-Validation Validation</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            <p style="font-size: 0.95rem; color: #334155; margin-top:0; font-weight: 500; line-height: 1.6;">
                These metrics were generated using <b>Spatial K-Fold Cross-Validation</b> (K=5). By clustering grid cells into 
                independent spatial groups using K-Means and holding out entire regions during validation, we prevent 
                <i>spatial autocorrelation leakage</i>, ensuring that model accuracies generalize to new regions.
            </p>
            """,
            unsafe_allow_html=True
        )
        
        # Summary metrics table
        metrics_summary = []
        for model_name, m_data in metrics.items():
            metrics_summary.append({
                "Model": model_name,
                "Spatial ROC-AUC": f"{m_data['roc_auc']:.4f}",
                "PR-AUC (Average Precision)": f"{m_data['pr_auc']:.4f}",
                "F1 Score": f"{m_data['f1']:.4f}",
                "Precision": f"{m_data['precision']:.4f}",
                "Recall": f"{m_data['recall']:.4f}"
            })
        st.table(pd.DataFrame(metrics_summary).set_index("Model"))
        
        # Plotly chart columns
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("<h4 style='font-size:1.05rem; margin-bottom:10px;'>Spatial ROC Curves (Receiver Operating Characteristic)</h4>", unsafe_allow_html=True)
            fig_roc = go.Figure()
            
            for model_name, m_data in metrics.items():
                y_true = np.array(m_data["y_true"])
                y_prob = np.array(m_data["y_prob"])
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                
                # Downsample curve points for smooth plotting
                if len(fpr) > 100:
                    indices = np.linspace(0, len(fpr) - 1, 100, dtype=int)
                    fpr, tpr = fpr[indices], tpr[indices]
                    
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, 
                    mode='lines', 
                    name=f"{model_name} (AUC: {m_data['roc_auc']:.3f})"
                ))
                
            fig_roc.add_shape(
                type="line", line=dict(dash='dash', color='#475569'),
                x0=0, x1=1, y0=0, y1=1
            )
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor='#e2e8f0', color='#334155', tickfont=dict(size=10)),
                yaxis=dict(gridcolor='#e2e8f0', color='#334155', tickfont=dict(size=10)),
                legend=dict(font=dict(color='#0f172a', size=10))
            )
            st.plotly_chart(fig_roc, use_container_width=True, config={'displayModeBar': False})
            
        with chart_col2:
            st.markdown("<h4 style='font-size:1.05rem; margin-bottom:10px;'>Precision-Recall Curves</h4>", unsafe_allow_html=True)
            fig_pr = go.Figure()
            
            for model_name, m_data in metrics.items():
                y_true = np.array(m_data["y_true"])
                y_prob = np.array(m_data["y_prob"])
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                
                if len(precision) > 100:
                    indices = np.linspace(0, len(precision) - 1, 100, dtype=int)
                    precision, recall = precision[indices], recall[indices]
                    
                fig_pr.add_trace(go.Scatter(
                    x=recall, y=precision, 
                    mode='lines', 
                    name=f"{model_name} (PR-AUC: {m_data['pr_auc']:.3f})"
                ))
                
            fig_pr.update_layout(
                xaxis_title="Recall",
                yaxis_title="Precision",
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor='#e2e8f0', color='#334155', tickfont=dict(size=10)),
                yaxis=dict(gridcolor='#e2e8f0', color='#334155', tickfont=dict(size=10)),
                legend=dict(font=dict(color='#0f172a', size=10))
            )
            st.plotly_chart(fig_pr, use_container_width=True, config={'displayModeBar': False})
            
        # Confusion Matrix render
        st.markdown("<br><h4 style='font-size:1.05rem; margin-bottom:10px;'>Confusion Matrix (XGBoost Classifier)</h4>", unsafe_allow_html=True)
        cm = metrics["XGBoost"]["confusion_matrix"]
        
        cm_col1, cm_col2 = st.columns([1, 2])
        with cm_col1:
            # Table-style display with light theme
            st.markdown(
                f"""
                <div style="background-color: #ffffff; border: 1px solid #cbd5e1; padding: 18px; border-radius: 8px; font-size: 0.95rem; color: #0f172a; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <table style="width: 100%; border-collapse: collapse; text-align: center;">
                        <tr style="border-bottom: 2px solid #cbd5e1;">
                            <th></th>
                            <th style="padding: 6px; color: #334155; font-weight:700;">Predicted Neg</th>
                            <th style="padding: 6px; color: #334155; font-weight:700;">Predicted Pos</th>
                        </tr>
                        <tr style="border-bottom: 1px solid #cbd5e1;">
                            <td style="padding: 8px; font-weight: bold; color: #334155; text-align: left;">Actual Neg</td>
                            <td style="background-color: rgba(5, 150, 105, 0.12); color: #047857; font-weight: bold;">{cm[0][0]}</td>
                            <td style="background-color: rgba(220, 38, 38, 0.08); color: #dc2626; font-weight: bold;">{cm[0][1]}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold; color: #334155; text-align: left;">Actual Pos</td>
                            <td style="background-color: rgba(220, 38, 38, 0.08); color: #dc2626; font-weight: bold;">{cm[1][0]}</td>
                            <td style="background-color: rgba(5, 150, 105, 0.22); color: #047857; font-weight: bold;">{cm[1][1]}</td>
                        </tr>
                    </table>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with cm_col2:
            st.markdown(
                """
                <div style="padding-left: 20px;">
                    <ul style="font-size: 0.95rem; color: #334155; line-height: 1.7; margin-top: 5px; font-weight: 500;">
                        <li><b>True Negatives ({0})</b>: Correctly identified cells where low-conflict matches environmental thresholds.</li>
                        <li><b>True Positives ({1})</b>: Successfully mapped high-conflict edge interfaces.</li>
                        <li><b>False Positives ({2})</b>: Conservation warning areas (predicted high risk but no historical records). Scientifically critical for precautionary forest patrols.</li>
                        <li><b>False Negatives ({3})</b>: Missed warning signs. Critical for local safety iterations.</li>
                    </ul>
                </div>
                """.format(cm[0][0], cm[1][1], cm[0][1], cm[1][0]),
                unsafe_allow_html=True
            )

    with tab_shap:
        st.markdown("<h3>Global Explainable AI (SHAP Summary)</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            <p style="font-size: 0.95rem; color: #334155; margin-top:0; font-weight: 500;">
                The chart below shows the global importance of each feature in predicting human-elephant conflict risk. 
                Importance is measured as the <b>mean absolute SHAP value</b> (in log-odds impact) across the landscape. 
                Features at the top represent the strongest environmental constraints on conflict risk.
            </p>
            """,
            unsafe_allow_html=True
        )
        
        # Calculate feature importances based on global SHAP values
        features_list = ["d_road", "d_water", "d_settlement", "d_forest", "forest_cover", "agricultural_exposure", "elevation", "slope"]
        X = df[features_list]
        
        # Compute SHAP values for the full dataset (very fast in TreeExplainer)
        shap_values = explainer(X)
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
        
        feature_names_clean = {
            "d_road": "Road Proximity (d_road)",
            "d_water": "Water Proximity (d_water)",
            "d_settlement": "Settlement Proximity (d_settlement)",
            "d_forest": "Forest Proximity (d_forest)",
            "forest_cover": "Forest Cover Density (%)",
            "agricultural_exposure": "Agricultural Exposure (%)",
            "elevation": "Elevation (meters)",
            "slope": "Slope (degrees)"
        }
        
        shap_imp_df = pd.DataFrame({
            'Feature': [feature_names_clean[f] for f in features_list],
            'Mean SHAP (Log-Odds Impact)': mean_abs_shap
        }).sort_values('Mean SHAP (Log-Odds Impact)', ascending=True)
        
        # Render Bar chart
        fig_shap_glob = go.Figure()
        fig_shap_glob.add_trace(go.Bar(
            y=shap_imp_df['Feature'],
            x=shap_imp_df['Mean SHAP (Log-Odds Impact)'],
            orientation='h',
            marker=dict(
                color='#059669',
                line=dict(color='#047857', width=1.5)
            )
        ))
        
        fig_shap_glob.update_layout(
            xaxis_title="Mean Absolute SHAP Value (Log-Odds Influence)",
            yaxis_title=None,
            margin=dict(l=20, r=20, t=10, b=20),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor='#e2e8f0', color='#334155', tickfont=dict(size=10)),
            yaxis=dict(color='#0f172a', tickfont=dict(size=10))
        )
        st.plotly_chart(fig_shap_glob, use_container_width=True, config={'displayModeBar': False})
        
        # Interpretations card
        st.markdown(
            """
            <div class="premium-card" style="border-color: #cbd5e1;">
                <h4 style="margin-top:0; color: #059669; font-size: 1.05rem; text-transform: uppercase; font-weight: 700;">
                    Interpretability Key Discoveries
                </h4>
                <p style="font-size: 0.95rem; color: #334155; margin: 5px 0 15px 0; line-height: 1.6; font-weight: 500;">
                    Analyzing the spatial variables through SHAP reveals key ecological drivers:
                </p>
                <ul style="font-size: 0.95rem; color: #0f172a; line-height: 1.7; padding-left: 20px; margin: 0; font-weight: 500;">
                    <li style="margin-bottom: 8px;">
                        <b>Agricultural Exposure & Forest Cover</b> have the highest global influence. This indicates that crop depredation is concentrated primarily along fragmented forest edges where agricultural estates (tea/coffee/paddy) meet forest cover.
                    </li>
                    <li style="margin-bottom: 8px;">
                        <b>Distance to Settlements</b> has a strong threshold effect: areas within 1-3 km of human habitation at the forest edge have high conflict log-odds.
                    </li>
                    <li style="margin-bottom: 0px;">
                        <b>Elevation & Slope</b> serve as geographic delimiters: conflict probability declines on steep mountain slopes (> 25°) where elephants rarely navigate, and peaks in low-to-mid elevation valley corridors.
                    </li>
                </ul>
            </div>
            """, 
            unsafe_allow_html=True
        )
