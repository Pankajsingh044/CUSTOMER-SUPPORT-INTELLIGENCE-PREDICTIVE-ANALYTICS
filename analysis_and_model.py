"""
=============================================================================
STANDALONE ANALYSIS & MACHINE LEARNING PIPELINE
Automated execution of Data Preparation, Visualizations, and Model Evaluation.
Saves all visual artifacts and metric summaries to the 'outputs/' directory.
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

# Sklearn modules
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

# Custom package modules
from src.data_loader import load_dataset, clean_and_prepare_data, get_kpi_summary


def run_full_pipeline():
    print("=" * 70)
    print("CUSTOMER SUPPORT ANALYTICS & PREDICTIVE MODELING PIPELINE")
    print("=" * 70)
    
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Load Data
    data_file = os.path.join("data", "customer_support_tickets.csv")
    if not os.path.exists(data_file):
        data_file = os.path.join("data", "cleaned_customer_support_tickets.csv")
        
    print(f"\n[1/5] Ingesting dataset from: {data_file}")
    raw_df = load_dataset(data_file)
    print(f" -> Raw shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    
    # 2. Clean & Preprocess
    print("\n[2/5] Executing Data Cleaning & Feature Engineering...")
    df, audit = clean_and_prepare_data(raw_df)
    kpis = get_kpi_summary(df)
    print(f" -> Total Tickets: {kpis['total_tickets']}")
    print(f" -> Resolution Rate: {kpis['resolution_rate']}%")
    print(f" -> Median Resolution Time: {kpis['avg_resolution_hours']} hours")
    print(f" -> Average CSAT: {kpis['avg_csat']}/5.0")
    print(f" -> Escalation Rate: {kpis['escalation_rate']}%")
    
    # Save cleaned dataset
    clean_csv_path = os.path.join("data", "cleaned_customer_support_tickets.csv")
    df.to_csv(clean_csv_path, index=False)
    print(f" -> Cleaned dataset saved to: {clean_csv_path}")
    
    # 3. Generate Publication Visualizations
    print("\n[3/5] Generating and saving analytical visualizations to 'outputs/'...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Visual 1: Scatter Plot
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    valid_scatter = df.dropna(subset=["Resolution_Time_Hours", "Response_Time_Hours"])
    cap = valid_scatter["Resolution_Time_Hours"].quantile(0.99)
    valid_scatter = valid_scatter[valid_scatter["Resolution_Time_Hours"] <= cap]
    
    scatter = ax.scatter(
        valid_scatter["Response_Time_Hours"],
        valid_scatter["Resolution_Time_Hours"],
        c=valid_scatter["Priority_Score"],
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="none",
        s=30
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Priority Score (1=Low, 4=Critical)")
    ax.set_title("Scatter Plot: Response Time vs. Resolution Time (Hours)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Response Time (Hours)", fontsize=11)
    ax.set_ylabel("Resolution Time (Hours)", fontsize=11)
    plt.tight_layout()
    fig.savefig("outputs/scatter_plot.png", dpi=300)
    plt.close(fig)
    print(" -> Saved outputs/scatter_plot.png")
    
    # Visual 2: Treemap (Squarified Bar representation)
    try:
        import squarify
        has_squarify = True
    except ImportError:
        has_squarify = False
        
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    prod_counts = df["Product Purchased"].value_counts()
    if has_squarify:
        colors = sns.color_palette("viridis", len(prod_counts))
        squarify.plot(sizes=prod_counts.values, label=prod_counts.index, color=colors, alpha=0.8, ax=ax, text_kwargs={'fontsize':9, 'weight':'bold'})
        ax.axis('off')
        ax.set_title("Treemap: Product Purchased Ticket Distribution", fontsize=14, fontweight="bold")
    else:
        sns.barplot(x=prod_counts.values, y=prod_counts.index, palette="mako", ax=ax)
        ax.set_title("Hierarchical Product Breakdown (Treemap Equivalent)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Ticket Volume")
    plt.tight_layout()
    fig.savefig("outputs/treemap.png", dpi=300)
    plt.close(fig)
    print(" -> Saved outputs/treemap.png")
    
    # Visual 3: Correlogram (Heatmap)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    num_cols = ["Customer Age", "Response_Time_Hours", "Resolution_Time_Hours", "Customer Satisfaction Rating", "Priority_Score", "Escalated", "Desc_Length"]
    corr = df[[c for c in num_cols if c in df.columns]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, linewidths=1, ax=ax)
    ax.set_title("Correlogram: Correlation Heatmap of Support Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig("outputs/correlogram.png", dpi=300)
    plt.close(fig)
    print(" -> Saved outputs/correlogram.png")
    
    # Visual 4: Mosaic Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    mosaic_df = df[["Ticket Priority", "Ticket Channel"]].dropna()
    mosaic(mosaic_df, ["Ticket Priority", "Ticket Channel"], ax=ax, title="Mosaic Plot: Ticket Priority vs Channel Proportion", gap=0.015)
    plt.tight_layout()
    fig.savefig("outputs/mosaic_plot.png", dpi=300)
    plt.close(fig)
    print(" -> Saved outputs/mosaic_plot.png")
    
    # 4. Train & Evaluate Machine Learning Models
    print("\n[4/5] Training Machine Learning Models & Computing Rigorous Metrics...")
    
    # Classification: Customer Satisfaction
    valid_csat = df.dropna(subset=["Customer Satisfaction Rating"]).copy()
    valid_csat["Target_Satisfied"] = (valid_csat["Customer Satisfaction Rating"] >= 4.0).astype(int)
    
    feature_num = ["Customer Age", "Response_Time_Hours", "Resolution_Time_Hours", "Desc_Length"]
    feature_cat = ["Ticket Type", "Ticket Priority", "Ticket Channel", "Customer Gender"]
    
    X = valid_csat[feature_num + feature_cat]
    y = valid_csat["Target_Satisfied"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_num),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), feature_cat)
        ]
    )
    
    # Random Forest Classifier
    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test)
    rf_proba = rf_pipe.predict_proba(X_test)[:, 1]
    
    # Logistic Regression
    lr_pipe = Pipeline([
        ("prep", preprocessor),
        ("lr", LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
    
    # Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv = cross_val_score(rf_pipe, X, y, cv=cv, scoring="accuracy")
    
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred, zero_division=0)
    rf_rec = recall_score(y_test, rf_pred, zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
    rf_auc = roc_auc_score(y_test, rf_proba)
    
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_prec = precision_score(y_test, lr_pred, zero_division=0)
    lr_rec = recall_score(y_test, lr_pred, zero_division=0)
    lr_f1 = f1_score(y_test, lr_pred, zero_division=0)
    lr_auc = roc_auc_score(y_test, lr_proba)
    
    print("\n" + "=" * 50)
    print("MODEL EVALUATION: CUSTOMER SATISFACTION CLASSIFIER")
    print("=" * 50)
    print(f"Random Forest Classifier Accuracy : {rf_acc * 100:.2f}%")
    print(f"Random Forest Precision          : {rf_prec:.4f}")
    print(f"Random Forest Recall             : {rf_rec:.4f}")
    print(f"Random Forest F1-Score           : {rf_f1:.4f}")
    print(f"Random Forest ROC-AUC            : {rf_auc:.4f}")
    print(f"5-Fold CV Mean Accuracy          : {rf_cv.mean() * 100:.2f}% (+/- {rf_cv.std() * 100:.2f}%)")
    print("-" * 50)
    print(f"Logistic Regression Accuracy     : {lr_acc * 100:.2f}%")
    print(f"Logistic Regression ROC-AUC      : {lr_auc:.4f}")
    print("-" * 50)
    print("\nDetailed Classification Report (Random Forest):")
    print(classification_report(y_test, rf_pred, target_names=["Dissatisfied/Neutral", "Satisfied"]))
    
    # Save Confusion Matrix figure
    cm = confusion_matrix(y_test, rf_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred: Not Satisfied", "Pred: Satisfied"],
                yticklabels=["Actual: Not Satisfied", "Actual: Satisfied"])
    ax.set_title("Confusion Matrix: Random Forest Classifier", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig("outputs/confusion_matrix.png", dpi=300)
    plt.close(fig)
    print(" -> Saved outputs/confusion_matrix.png")
    
    # Save Feature Importance figure
    cat_enc = rf_pipe.named_steps["prep"].named_transformers_["cat"]
    encoded_names = list(cat_enc.get_feature_names_out(feature_cat))
    feat_names = feature_num + encoded_names
    importances = rf_pipe.named_steps["rf"].feature_importances_
    
    feat_series = pd.Series(importances, index=feat_names).sort_values(ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    feat_series.plot(kind="barh", color="#1f77b4", ax=ax)
    ax.set_title("Top 10 Feature Importances: Customer Satisfaction Drivers", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Importance Score")
    plt.tight_layout()
    fig.savefig("outputs/feature_importance.png", dpi=300)
    plt.close(fig)
    print(" -> Saved outputs/feature_importance.png")
    
    # Regression: Ticket Resolution Time
    valid_reg = df.dropna(subset=["Resolution_Time_Hours"]).copy()
    X_reg = valid_reg[feature_num[:3] + feature_cat[:3]]
    y_reg = valid_reg["Resolution_Time_Hours"]
    
    prep_reg = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_num[:3]),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), feature_cat[:3])
        ]
    )
    reg_pipe = Pipeline([("prep", prep_reg), ("rf", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42))])
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    reg_pipe.fit(X_r_train, y_r_train)
    y_r_pred = reg_pipe.predict(X_r_test)
    
    mae = mean_absolute_error(y_r_test, y_r_pred)
    rmse = np.sqrt(mean_squared_error(y_r_test, y_r_pred))
    r2 = r2_score(y_r_test, y_r_pred)
    
    print("\n" + "=" * 50)
    print("MODEL EVALUATION: RESOLUTION TIME REGRESSOR")
    print("=" * 50)
    print(f"Mean Absolute Error (MAE)        : {mae:.2f} hours")
    print(f"Root Mean Squared Error (RMSE)   : {rmse:.2f} hours")
    print(f"R² Determination Score           : {r2:.4f}")
    
    # 5. Export Metrics Summary to JSON
    metrics_summary = {
        "kpis": kpis,
        "classification": {
            "random_forest": {
                "accuracy": round(rf_acc, 4),
                "precision": round(rf_prec, 4),
                "recall": round(rf_rec, 4),
                "f1_score": round(rf_f1, 4),
                "roc_auc": round(rf_auc, 4),
                "cv_mean_accuracy": round(float(rf_cv.mean()), 4),
                "cv_std": round(float(rf_cv.std()), 4)
            },
            "logistic_regression": {
                "accuracy": round(lr_acc, 4),
                "precision": round(lr_prec, 4),
                "recall": round(lr_rec, 4),
                "f1_score": round(lr_f1, 4),
                "roc_auc": round(lr_auc, 4)
            }
        },
        "regression": {
            "mae_hours": round(float(mae), 2),
            "rmse_hours": round(float(rmse), 2),
            "r2_score": round(float(r2), 4)
        }
    }
    
    with open("outputs/evaluation_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(" -> Saved outputs/evaluation_metrics.json")
    
    print("\n[5/5] Pipeline successfully completed! All deliverables generated.")
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
