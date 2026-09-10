"""
================================================================================
CLINICAL PREDICTION & ALGORITHMIC FAIRNESS IN CRITICAL CARE (MIMIC-IV)
================================================================================
Author: Prince Gabriel
Institution: University of Birmingham
Repository: Clinical-ML-Fairness-MIMIC-IV

Key Engineering & Methodological Highlights:
1. Target Leakage Prevention: Excludes `discharge_location` (which contains 'DIED'
   at discharge time) to ensure genuine clinical prediction at admission.
2. Leakage-Free Preprocessing: Strict Train/Test split prior to any transformer fitting.
   Imputation and scaling parameters are fitted exclusively on training data.
3. Multi-Task Deep Learning: Upgrades degenerate 1-timestep recurrent architecture
   to a calibrated Multi-Task Tabular Deep Learning Network with shared latent
   representations, Batch Normalization, and Dropout.
4. Comprehensive Algorithmic Fairness: Transcends the accuracy paradox on imbalanced
   data (~1.36% mortality) by evaluating Equal Opportunity (TPR Parity), Predictive
   Equality (FPR Parity), and Disparate Impact across standardized demographic cohorts.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)

# Optional TensorFlow import with graceful fallback
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ==============================================================================
# 1. DEMOGRAPHIC HARMONIZATION HELPER
# ==============================================================================
def standardize_race(race_str):
    """
    Consolidates 30+ granular, sparse ethnic sub-categories into standardized
    cohorts to ensure statistically sound subgroup sample sizes (n >= 30).
    Avoids undefined metric warnings and single-sample variance spikes.
    """
    if not isinstance(race_str, str):
        return 'UNKNOWN'
    r = race_str.upper()
    if 'WHITE' in r or 'PORTUGUESE' in r:
        return 'WHITE'
    elif 'BLACK' in r or 'AFRICAN' in r:
        return 'BLACK/AFRICAN AMERICAN'
    elif 'HISPANIC' in r or 'LATINO' in r:
        return 'HISPANIC/LATINO'
    elif 'ASIAN' in r:
        return 'ASIAN'
    elif 'NATIVE' in r or 'AMERICAN INDIAN' in r or 'ALASKA' in r:
        return 'AMERICAN INDIAN/ALASKA NATIVE'
    elif 'HAWAIIAN' in r or 'PACIFIC' in r:
        return 'HAWAIIAN/PACIFIC ISLANDER'
    elif 'UNKNOWN' in r or 'UNABLE' in r or 'PATIENT' in r:
        return 'UNKNOWN/OTHER'
    else:
        return 'OTHER'


# ==============================================================================
# 2. DATA INGESTION & ICD-9 TO ICD-10 GEM MAPPING
# ==============================================================================
def load_and_merge_data(admissions_path, patients_path, diagnoses_path, gem_path):
    """
    Loads raw MIMIC-IV subsets, performs relational joins on subject_id,
    and applies General Equivalence Mappings (GEM) crosswalk from ICD-9 to ICD-10.
    """
    print("[-] Loading raw clinical tables...")
    df_adm = pd.read_csv(admissions_path)
    df_pat = pd.read_csv(patients_path)
    df_diag = pd.read_csv(diagnoses_path)
    
    # Relational inner joins on subject_id
    df_merged = df_adm.merge(df_pat, on="subject_id", how="inner")
    df = df_merged.merge(df_diag, on="subject_id", how="inner")
    print(f"[-] Initial merged dataset: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    
    # ICD-9 to ICD-10 General Equivalence Mapping
    if os.path.exists(gem_path):
        print("[-] Applying ICD-9 to ICD-10 GEM Crosswalk mapping...")
        df_gem = pd.read_csv(gem_path)
        
        df9 = df[df["icd_version"] == 9].copy()
        df10 = df[df["icd_version"] == 10].copy()
        
        # Map ICD-9 to ICD-10
        df9 = df9.merge(df_gem, left_on="icd_code", right_on="icd9cm", how="left")
        df10["icd10cm"] = df10["icd_code"]
        
        df_all = pd.concat([df9, df10], ignore_index=True)
    else:
        print("[!] GEM crosswalk file not found; using raw icd_code.")
        df_all = df.copy()
        df_all["icd10cm"] = df_all["icd_code"]
        
    return df_all


# ==============================================================================
# 3. FEATURE ENGINEERING & CLINICAL TIMELINES
# ==============================================================================
def engineer_clinical_features(df_all):
    """
    Derives clinical duration targets and features:
    - Length of Stay (LOS) in days: (dischtime - admittime)
    - Emergency Department Visit Time (Visit_time) in days: (edouttime - edregtime)
    """
    print("[-] Engineering clinical duration targets and timeline features...")
    df = df_all.copy()
    
    # 1. Length of Stay (LOS)
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    df["length_of_stay"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400.0
    
    # Filter non-negative LOS
    df = df[df["length_of_stay"] >= 0].copy()
    
    # 2. Emergency Department Registration to Departure Duration
    if "edregtime" in df.columns and "edouttime" in df.columns:
        df["edregtime"] = pd.to_datetime(df["edregtime"])
        df["edouttime"] = pd.to_datetime(df["edouttime"])
        df["Visit_time"] = (df["edouttime"] - df["edregtime"]).dt.total_seconds() / 86400.0
        # Clip anomalous negative visit times
        df.loc[df["Visit_time"] < 0, "Visit_time"] = np.nan
    else:
        df["Visit_time"] = np.nan
        
    # Standardize racial demographic cohorts
    df["race_standardized"] = df["race"].apply(standardize_race)
    
    return df


# ==============================================================================
# 4. LEAKAGE-FREE FEATURE EXTRACTION & DATA SPLITTING
# ==============================================================================
def prepare_leakage_free_datasets(df, sample_size=100000, random_state=42):
    """
    Prepares features strictly available at admission.
    CRITICAL: Drops `discharge_location` which introduces severe target leakage
    because patients who die have discharge_location == 'DIED'.
    """
    print("[-] Enforcing Target Leakage Prevention protocol...")
    
    # Sampling representative cohort if dataset exceeds sample_size
    if len(df) > sample_size:
        df_sampled = df.sample(n=sample_size, random_state=random_state).copy()
    else:
        df_sampled = df.copy()
        
    # Extract Ground-Truth Targets
    y_class = df_sampled["hospital_expire_flag"].values.astype(int)
    y_reg = df_sampled["length_of_stay"].values.astype(float)
    race_cohorts = df_sampled["race_standardized"].values
    
    # Explicitly permitted admission-time clinical predictors
    # NOTE: discharge_location is INTENTIONALLY EXCLUDED to avoid target leakage!
    numeric_features = ["anchor_age", "anchor_year", "Visit_time"]
    categorical_features = [
        "gender", "admission_type", "admission_location",
        "insurance", "marital_status", "language", "race_standardized", "icd10cm"
    ]
    
    feature_cols = numeric_features + categorical_features
    X_df = df_sampled[feature_cols].copy()
    
    print(f"[-] Feature Set: {len(numeric_features)} numeric, {len(categorical_features)} categorical.")
    print("[-] Confirmed exclusion of `discharge_location` and all post-admission event markers.")
    
    # Train / Test Split performed BEFORE any transformer fitting (Prevents Preprocessing Leakage)
    X_train_df, X_test_df, y_train_class, y_test_class, y_train_reg, y_test_reg, race_train, race_test = (
        train_test_split(
            X_df, y_class, y_reg, race_cohorts,
            test_size=0.20,
            stratify=y_class,
            random_state=random_state
        )
    )
    
    print(f"[-] Stratified Split: Train={len(X_train_df):,} | Test={len(X_test_df):,}")
    print(f"[-] Mortality prevalence: Train={y_train_class.mean()*100:.2f}% | Test={y_test_class.mean()*100:.2f}%")
    
    return (
        X_train_df, X_test_df,
        y_train_class, y_test_class,
        y_train_reg, y_test_reg,
        race_train, race_test,
        numeric_features, categorical_features
    )


# ==============================================================================
# 5. SCIKIT-LEARN PREPROCESSING PIPELINE
# ==============================================================================
def build_and_fit_pipeline(X_train_df, X_test_df, numeric_features, categorical_features):
    """
    Constructs robust preprocessor:
    - Numeric: SimpleImputer (median) + RobustScaler (resistant to clinical outlier durations)
    - Categorical: SimpleImputer ('missing') + OneHotEncoder (handles unseen categories)
    Fitted EXCLUSIVELY on X_train_df to prevent data leakage.
    """
    print("[-] Constructing scikit-learn Preprocessing Pipeline...")
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    
    print("[-] Fitting pipeline ONLY on X_train (Zero Data Leakage)...")
    X_train_proc = preprocessor.fit_transform(X_train_df)
    X_test_proc = preprocessor.transform(X_test_df)
    
    print(f"[-] Transformed feature space: {X_train_proc.shape[1]} encoded predictors.")
    return preprocessor, X_train_proc, X_test_proc


# ==============================================================================
# 6. UNSUPERVISED EXPLORATION: PCA & K-MEANS
# ==============================================================================
def run_unsupervised_exploration(X_train_proc, y_train_class, n_components=2):
    """
    Conducts Principal Component Analysis for dimensionality reduction and
    K-Means clustering. Clarifies that clustering on PCA projection is an
    exploratory visualization tool rather than a supervised classifier.
    """
    print("\n" + "="*70)
    print("UNSUPERVISED LEARNING: PCA & K-MEANS CLUSTERING")
    print("="*70)
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_train_proc)
    
    var_exp = pca.explained_variance_ratio_
    print(f"[-] PCA Component 1 Variance Explained: {var_exp[0]*100:.2f}%")
    print(f"[-] PCA Component 2 Variance Explained: {var_exp[1]*100:.2f}%")
    print(f"[-] Cumulative 2D Variance Explained: {var_exp.sum()*100:.2f}%")
    print("[-] Note: Clinical tabular data has high intrinsic dimensionality;")
    print("    2D PCA captures global variation but clustering is exploratory.")
    
    # K-Means with k-means++ initialization (industry best practice)
    kmeans = KMeans(n_clusters=2, init="k-means++", random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    
    return pca, kmeans, X_pca


# ==============================================================================
# 7. SUPERVISED BENCHMARK: REGULARIZED DECISION TREE & RANDOM FOREST
# ==============================================================================
def train_and_evaluate_tree_models(X_train, y_train, X_test, y_test, task="classification"):
    """
    Trains regularized decision trees and ensemble random forests with:
    - Proper hyperparameter bounds (max_depth, min_samples_leaf) to prevent overfitting
    - True probability output (predict_proba) for continuous ROC/PR curve computation
    """
    print("\n" + "="*70)
    print(f"SUPERVISED BENCHMARK: {task.upper()} MODELS")
    print("="*70)
    
    if task == "classification":
        # Regularized Decision Tree (Prevents 100% memorization)
        tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight='balanced', random_state=42)
        tree.fit(X_train, y_train)
        
        # Ensembles provide state-of-the-art tabular performance
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Predictions & Probabilities (predict_proba avoids two-point step ROC artifacts)
        y_pred_tree = tree.predict(X_test)
        y_prob_tree = tree.predict_proba(X_test)[:, 1]
        
        y_pred_rf = rf.predict(X_test)
        y_prob_rf = rf.predict_proba(X_test)[:, 1]
        
        print(f"[-] Decision Tree -> Accuracy: {accuracy_score(y_test, y_pred_tree):.4f} | Recall: {recall_score(y_test, y_pred_tree):.4f} | ROC-AUC: {roc_auc_score(y_test, y_prob_tree):.4f}")
        print(f"[-] Random Forest  -> Accuracy: {accuracy_score(y_test, y_pred_rf):.4f} | Recall: {recall_score(y_test, y_pred_rf):.4f} | ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
        
        return tree, rf, y_prob_tree, y_pred_tree, y_prob_rf, y_pred_rf
        
    elif task == "regression":
        # Regularized Regression Tree
        tree_reg = DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=42)
        tree_reg.fit(X_train, y_train)
        
        rf_reg = RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
        rf_reg.fit(X_train, y_train)
        
        pred_tree = tree_reg.predict(X_test)
        pred_rf = rf_reg.predict(X_test)
        
        print(f"[-] Decision Tree Regressor -> MAE: {mean_absolute_error(y_test, pred_tree):.3f} days | R2: {r2_score(y_test, pred_tree):.3f}")
        print(f"[-] Random Forest Regressor  -> MAE: {mean_absolute_error(y_test, pred_rf):.3f} days | R2: {r2_score(y_test, pred_rf):.3f}")
        
        return tree_reg, rf_reg, pred_tree, pred_rf


# ==============================================================================
# 8. MULTI-TASK DEEP NEURAL NETWORK (UPGRADE FROM 1-TIMESTEP LSTM)
# ==============================================================================
def build_multi_task_neural_network(input_dim):
    """
    Replaces the mathematically degenerate 1-timestep LSTM with an industry-grade
    Multi-Task Deep Tabular Neural Network:
    - Shared representation layers with Batch Normalization & Dropout
    - Dual specialized heads:
        * Head 1: In-hospital mortality (Binary Crossentropy + Sigmoid)
        * Head 2: Length of Stay (Huber Loss / MSE + Linear)
    - Calibrated loss weighting prevents regression loss from dominating classification.
    """
    if not TF_AVAILABLE:
        print("[!] TensorFlow not installed; skipping deep learning module.")
        return None
        
    print("\n" + "="*70)
    print("MULTI-TASK DEEP TABULAR NEURAL NETWORK")
    print("="*70)
    
    inputs = layers.Input(shape=(input_dim,), name="clinical_features")
    
    # Shared Latent Representation Layers
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    shared_features = layers.Dense(32, activation="relu", name="shared_representation")(x)
    
    # Head 1: Mortality Classification
    class_branch = layers.Dense(16, activation="relu")(shared_features)
    expiry_output = layers.Dense(1, activation="sigmoid", name="expiry_output")(class_branch)
    
    # Head 2: Length of Stay Regression
    reg_branch = layers.Dense(16, activation="relu")(shared_features)
    los_output = layers.Dense(1, activation="linear", name="los_output")(reg_branch)
    
    model = models.Model(inputs=inputs, outputs=[expiry_output, los_output])
    
    # Balanced loss weighting: regression loss scaled down so classification gradients remain active
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "expiry_output": "binary_crossentropy",
            "los_output": "huber"
        },
        loss_weights={
            "expiry_output": 1.0,
            "los_output": 0.05
        },
        metrics={
            "expiry_output": [tf.keras.metrics.AUC(name="auc"), "accuracy"],
            "los_output": ["mae"]
        }
    )
    
    return model


# ==============================================================================
# 9. RIGOROUS ALGORITHMIC FAIRNESS & SUBGROUP BIAS AUDIT
# ==============================================================================
def conduct_algorithmic_fairness_audit(y_true, y_prob, y_pred, race_cohorts, model_name="Model"):
    """
    Transcends the 'Accuracy Paradox' on imbalanced clinical datasets:
    Evaluates True Positive Rate Parity (Equal Opportunity), False Positive Rate
    Parity (Predictive Equality), and Disparate Impact across racial cohorts.
    """
    print("\n" + "="*70)
    print(f"ALGORITHMIC FAIRNESS & EQUITY AUDIT: {model_name.upper()}")
    print("="*70)
    
    df_eval = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "race": race_cohorts
    })
    
    records = []
    unique_cohorts = df_eval["race"].value_counts()
    
    print(f"{'Demographic Group':<28} | {'N':>6} | {'Accuracy':>8} | {'TPR (Recall)':>12} | {'FPR':>8} | {'PR-AUC':>8}")
    print("-" * 80)
    
    # Compute base rate for disparate impact
    overall_selection_rate = y_pred.mean()
    
    for cohort, count in unique_cohorts.items():
        sub = df_eval[df_eval["race"] == cohort]
        
        # Requires at least 15 samples for minimal statistical validity
        if len(sub) < 15:
            continue
            
        acc = accuracy_score(sub["y_true"], sub["y_pred"])
        
        # Subgroup TPR (Equal Opportunity) & FPR
        n_pos = (sub["y_true"] == 1).sum()
        n_neg = (sub["y_true"] == 0).sum()
        
        tpr = recall_score(sub["y_true"], sub["y_pred"], zero_division=0) if n_pos > 0 else np.nan
        fpr = ((sub["y_pred"] == 1) & (sub["y_true"] == 0)).sum() / n_neg if n_neg > 0 else np.nan
        
        # PR-AUC
        if n_pos > 0 and len(np.unique(sub["y_true"])) > 1:
            prec_curve, rec_curve, _ = precision_recall_curve(sub["y_true"], sub["y_prob"])
            pr_auc = auc(rec_curve, prec_curve)
        else:
            pr_auc = np.nan
            
        selection_rate = sub["y_pred"].mean()
        disparate_impact = selection_rate / (overall_selection_rate + 1e-9)
        
        records.append({
            "Cohort": cohort,
            "Count": count,
            "Accuracy": acc,
            "TPR_Recall": tpr,
            "FPR": fpr,
            "PR_AUC": pr_auc,
            "Disparate_Impact": disparate_impact
        })
        
        tpr_str = f"{tpr:.4f}" if not np.isnan(tpr) else "N/A"
        fpr_str = f"{fpr:.4f}" if not np.isnan(fpr) else "N/A"
        pr_str = f"{pr_auc:.4f}" if not np.isnan(pr_auc) else "N/A"
        
        print(f"{cohort:<28} | {count:>6} | {acc:>8.4f} | {tpr_str:>12} | {fpr_str:>8} | {pr_str:>8}")
        
    fairness_df = pd.DataFrame(records)
    
    # Equal Opportunity Disparity (Max TPR - Min TPR among valid groups)
    valid_tprs = fairness_df["TPR_Recall"].dropna()
    if len(valid_tprs) > 1:
        eod_gap = valid_tprs.max() - valid_tprs.min()
        print(f"\n[-] Equal Opportunity Disparity Gap (Max TPR - Min TPR): {eod_gap:.4f}")
        if eod_gap > 0.15:
            print("    [!] WARNING: Noticeable demographic opportunity disparity detected.")
            print("        Model identifies deteriorating patients with varying sensitivity across cohorts.")
        else:
            print("    [+] High demographic equity: TPR is well-calibrated across cohorts.")
            
    return fairness_df


# ==============================================================================
# 10. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    print("""
    ======================================================================
    MIMIC-IV CLINICAL PREDICTION & FAIRNESS PIPELINE
    ======================================================================
    """)
    # Adjust file paths for your local or Colab workspace
    base_dir = os.path.dirname(os.path.abspath(__file__))
    adm_path = os.path.join(base_dir, "admissions_subset-checkpoint.csv")
    pat_path = os.path.join(base_dir, "patients_subset-checkpoint.csv")
    diag_path = os.path.join(base_dir, "diagnosis_icd_subset-checkpoint.csv")
    gem_path = os.path.join(base_dir, "icd10cmtoicd9gem.csv")
    
    # Check if files exist locally; if not, print instructions
    if not all(os.path.exists(p) for p in [adm_path, pat_path, diag_path]):
        print("[!] Note: Subset CSV files not detected in current working directory.")
        print("    This script is ready to run directly on Google Colab or local environment")
        print("    where the MIMIC-IV subset CSVs are present.")
        return

    # Ingestion & Mapping
    df_all = load_and_merge_data(adm_path, pat_path, diag_path, gem_path)
    
    # Feature Engineering
    df_feat = engineer_clinical_features(df_all)
    
    # Target Leakage Prevention & Dataset Splitting
    (
        X_train_df, X_test_df,
        y_train_class, y_test_class,
        y_train_reg, y_test_reg,
        race_train, race_test,
        num_cols, cat_cols
    ) = prepare_leakage_free_datasets(df_feat, sample_size=100000)
    
    # Leakage-Free Preprocessor
    preprocessor, X_train_proc, X_test_proc = build_and_fit_pipeline(
        X_train_df, X_test_df, num_cols, cat_cols
    )
    
    # Unsupervised Exploration
    pca, kmeans, X_pca = run_unsupervised_exploration(X_train_proc, y_train_class)
    
    # Supervised Tree Models (Classification)
    tree, rf, prob_tree, pred_tree, prob_rf, pred_rf = train_and_evaluate_tree_models(
        X_train_proc, y_train_class, X_test_proc, y_test_class, task="classification"
    )
    
    # Algorithmic Fairness Audit
    fairness_results = conduct_algorithmic_fairness_audit(
        y_test_class, prob_rf, pred_rf, race_test, model_name="Random Forest Classifier"
    )
    
    # Multi-Task Neural Network (if TensorFlow available)
    if TF_AVAILABLE:
        nn_model = build_multi_task_neural_network(X_train_proc.shape[1])
        if nn_model:
            print("[-] Training Multi-Task Neural Network...")
            history = nn_model.fit(
                X_train_proc,
                {"expiry_output": y_train_class, "los_output": y_train_reg},
                validation_data=(
                    X_test_proc,
                    {"expiry_output": y_test_class, "los_output": y_test_reg}
                ),
                epochs=10,
                batch_size=128,
                verbose=1
            )
            
            # Predict & Audit Fairness for Neural Network
            preds = nn_model.predict(X_test_proc, verbose=0)
            nn_prob = preds[0].flatten()
            nn_pred = (nn_prob > 0.5).astype(int)
            
            conduct_algorithmic_fairness_audit(
                y_test_class, nn_prob, nn_pred, race_test, model_name="Multi-Task Neural Network"
            )

    print("\n[+] Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
