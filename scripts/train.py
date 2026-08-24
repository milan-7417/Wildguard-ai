import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    f1_score, precision_score, recall_score, confusion_matrix
)
import xgboost as xgb
import shap

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = [
    "d_road", "d_water", "d_settlement", "d_forest", 
    "forest_cover", "agricultural_exposure", "elevation", "slope"
]

def train_and_evaluate():
    print("Loading data for training...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "model_features.csv"))
    
    X = df[FEATURES]
    y = df["target"]
    coords = df[["centroid_lat", "centroid_lon"]]
    
    # 1. Geographically Aware Spatial K-Fold Split
    # We cluster the grid cells into 5 spatial zones based on coordinates to prevent spatial leakage
    print("Performing K-Means spatial clustering for cross-validation...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["spatial_cluster"] = kmeans.fit_predict(coords)
    
    clusters = df["spatial_cluster"].unique()
    
    # Store CV evaluation metrics
    xgb_cv_preds = np.zeros(len(df))
    rf_cv_preds = np.zeros(len(df))
    lr_cv_preds = np.zeros(len(df))
    
    xgb_cv_probs = np.zeros(len(df))
    rf_cv_probs = np.zeros(len(df))
    lr_cv_probs = np.zeros(len(df))
    
    for fold, val_cluster in enumerate(clusters):
        print(f"--- Spatial CV Fold {fold+1}/5 (Holding out Cluster {val_cluster}) ---")
        train_idx = df["spatial_cluster"] != val_cluster
        val_idx = df["spatial_cluster"] == val_cluster
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, 
            random_state=42, eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        xgb_cv_probs[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        xgb_cv_preds[val_idx] = xgb_model.predict(X_val)
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_cv_probs[val_idx] = rf_model.predict_proba(X_val)[:, 1]
        rf_cv_preds[val_idx] = rf_model.predict(X_val)
        
        # Logistic Regression
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        lr_cv_probs[val_idx] = lr_model.predict_proba(X_val)[:, 1]
        lr_cv_preds[val_idx] = lr_model.predict(X_val)

    # Evaluate Models
    metrics = {}
    for name, probs, preds in [("XGBoost", xgb_cv_probs, xgb_cv_preds), 
                               ("Random Forest", rf_cv_probs, rf_cv_preds), 
                               ("Logistic Regression", lr_cv_probs, lr_cv_preds)]:
        roc_auc = roc_auc_score(y, probs)
        precision, recall, _ = precision_recall_curve(y, probs)
        pr_auc = auc(recall, precision)
        f1 = f1_score(y, preds)
        prec = precision_score(y, preds)
        rec = recall_score(y, preds)
        cm = confusion_matrix(y, preds).tolist()
        
        print(f"\n{name} CV Metrics:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC:  {pr_auc:.4f}")
        print(f"  F1:      {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        
        # Store for serialization (curves lists converted to list for JSON compatibility)
        metrics[name] = {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "confusion_matrix": cm,
            "y_true": y.tolist(),
            "y_prob": probs.tolist()
        }

    # Save metrics JSON
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved models/metrics.json")
    
    # 2. Train final XGBoost model on the complete dataset
    print("\nTraining final XGBoost model on complete dataset...")
    final_xgb = xgb.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.05, 
        random_state=42, eval_metric='logloss'
    )
    final_xgb.fit(X, y)
    
    # Save Model
    # We save as booster json for native loading in XGBoost
    final_xgb.save_model(os.path.join(MODELS_DIR, "xgb_model.json"))
    print("Saved final XGBoost model to models/xgb_model.json")
    
    # 3. Create and Save SHAP Explainer
    print("Fitting and saving SHAP explainer...")
    explainer = shap.TreeExplainer(final_xgb)
    
    # We save the explainer and background dataset using pickle
    # Streamlit can load this to generate SHAP values instantly
    with open(os.path.join(MODELS_DIR, "shap_explainer.pkl"), "wb") as f:
        pickle.dump(explainer, f)
    
    # Also save the features df for easy model predictions in the UI
    df.to_csv(os.path.join(PROCESSED_DIR, "final_features_with_clusters.csv"), index=False)
    print("Saved models/shap_explainer.pkl and updated final features.")

if __name__ == "__main__":
    train_and_evaluate()
