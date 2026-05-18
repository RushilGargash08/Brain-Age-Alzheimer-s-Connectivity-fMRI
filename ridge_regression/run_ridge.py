import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "/Users/rushilgargash/Desktop/mlpr/numpy_extracted"
ADNIMERGE_PATH = "/Users/rushilgargash/Desktop/mlpr/ADNIMERGE_23Apr2026.csv"
OUTPUT_DIR = "/Users/rushilgargash/Desktop/mlpr/ridge_regression/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """ Load subject data, extract correct hemispheric features, and apply Fisher z-transform. """
    print("Loading ADNIMERGE to extract Baseline Ages...")
    df = pd.read_csv(ADNIMERGE_PATH, usecols=['PTID', 'VISCODE', 'AGE', 'DX_bl'])
    df_bl = df[df['VISCODE'] == 'bl'].drop_duplicates(subset=['PTID'])
    df_bl['clean_id'] = df_bl['PTID'].str.replace('_', '')
    
    valid_subjects = []
    features_ll, features_rr, features_lr = [], [], []
    
    files = os.listdir(DATA_DIR)
    subject_ids = sorted(list(set([f.split('_')[0] for f in files if f.endswith('.npy')])))
    print(f"Found {len(subject_ids)} unique subjects in {DATA_DIR}")
    
    for sub_id in subject_ids:
        sub_meta = df_bl[df_bl['clean_id'] == sub_id]
        if len(sub_meta) == 0:
            continue
            
        age = sub_meta.iloc[0]['AGE']
        dx = sub_meta.iloc[0]['DX_bl']
        
        if pd.isna(age):
            continue
            
        paths = [os.path.join(DATA_DIR, f"{sub_id}_{h}.npy") for h in ['ll', 'rr', 'lr']]
        if all(os.path.exists(p) for p in paths):
            ll = np.load(paths[0])
            rr = np.load(paths[1])
            lr = np.load(paths[2])
            
            # Extract LL upper, RR upper, LR full
            ll_feat = ll[np.triu_indices(ll.shape[0], k=1)] if ll.ndim == 2 else ll
            rr_feat = rr[np.triu_indices(rr.shape[0], k=1)] if rr.ndim == 2 else rr
            lr_feat = lr.flatten()
            
            # Fisher z-transform: z = 0.5 * ln((1+r)/(1-r))
            ll_z = np.arctanh(np.clip(ll_feat, -0.9999, 0.9999))
            rr_z = np.arctanh(np.clip(rr_feat, -0.9999, 0.9999))
            lr_z = np.arctanh(np.clip(lr_feat, -0.9999, 0.9999))
            
            features_ll.append(ll_z)
            features_rr.append(rr_z)
            features_lr.append(lr_z)
            
            valid_subjects.append({
                'subject_id': sub_id,
                'chronological_age': age,
                'dx': dx
            })
            
    df_valid = pd.DataFrame(valid_subjects)
    return df_valid, np.array(features_ll), np.array(features_rr), np.array(features_lr)

def train_normative_and_predict(X_cn, y_cn, X_ad, name):
    """
    1. Evaluates healthy controls (CN) using LOOCV (completely unbiased).
    2. Performs LOOCV-based age-residualization of BAG for healthy controls to prevent leakage.
    3. Trains a final model on all CN subjects to predict AD patients and residualizes their BAG.
    """
    # 1. Leave-One-Out CV for CN Subjects
    loo = LeaveOneOut()
    cn_preds_raw = np.zeros(len(X_cn))
    cn_preds_corr = np.zeros(len(X_cn))
    
    for train_idx, test_idx in loo.split(X_cn):
        X_tr, X_te = X_cn[train_idx], X_cn[test_idx]
        y_tr, y_te = y_cn[train_idx], y_cn[test_idx]
        
        # Feature Selection
        selector = VarianceThreshold(threshold=0.0001)
        X_tr_sel = selector.fit_transform(X_tr)
        X_te_sel = selector.transform(X_te)
        
        # Scale
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_sel)
        X_te_scaled = scaler.transform(X_te_sel)
        
        # PCA
        pca = PCA(n_components=0.90, svd_solver='full', random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_scaled)
        X_te_pca = pca.transform(X_te_scaled)
        
        # RidgeCV Model
        ridge = RidgeCV(alphas=np.logspace(-1.5, 1.5, 40))
        ridge.fit(X_tr_pca, y_tr)
        pred_raw = ridge.predict(X_te_pca)[0]
        
        # In-fold predictions on training set for bias-correction estimation
        pred_tr_raw = ridge.predict(X_tr_pca)
        bags_tr_raw = pred_tr_raw - y_tr
        
        # Fit Linear Regression on training controls
        lr_bias = LinearRegression()
        lr_bias.fit(y_tr.reshape(-1, 1), bags_tr_raw)
        
        # Predict expected bias and residualize held-out subject BAG
        expected_bias = lr_bias.predict(y_te.reshape(-1, 1))[0]
        bag_raw = pred_raw - y_te[0]
        bag_res = bag_raw - expected_bias
        
        cn_preds_raw[test_idx[0]] = pred_raw
        cn_preds_corr[test_idx[0]] = y_te[0] + bag_res
        
    # 2. Train on ALL CN Subjects to predict AD Patients
    selector_final = VarianceThreshold(threshold=0.0001)
    X_cn_sel = selector_final.fit_transform(X_cn)
    X_ad_sel = selector_final.transform(X_ad)
    
    scaler_final = StandardScaler()
    X_cn_scaled = scaler_final.fit_transform(X_cn_sel)
    X_ad_scaled = scaler_final.transform(X_ad_sel)
    
    pca_final = PCA(n_components=0.90, svd_solver='full', random_state=42)
    X_cn_pca = pca_final.fit_transform(X_cn_scaled)
    X_ad_pca = pca_final.transform(X_ad_scaled)
    
    ridge_final = RidgeCV(alphas=np.logspace(-1.5, 1.5, 40))
    ridge_final.fit(X_cn_pca, y_cn)
    
    ad_preds_raw = ridge_final.predict(X_ad_pca)
    ad_bags_raw = ad_preds_raw - y_ad
    
    # Fit final bias correction on all CN subjects using LOOCV raw bags
    cn_bags_raw = cn_preds_raw - y_cn
    lr_final = LinearRegression()
    lr_final.fit(y_cn.reshape(-1, 1), cn_bags_raw)
    expected_bias_ad = lr_final.predict(y_ad.reshape(-1, 1))
    
    ad_bags_residual = ad_bags_raw - expected_bias_ad
    ad_preds_corr = y_ad + ad_bags_residual
    
    # Evaluate CN metrics
    mae_cn = mean_absolute_error(y_cn, cn_preds_raw)
    r2_cn = r2_score(y_cn, cn_preds_raw)
    r_pearson, p_pearson = pearsonr(y_cn, cn_preds_raw)
    
    print(f"\nModel: {name}")
    print(f"  Healthy Control (CN) Age Prediction Performance (LOOCV)")
    print(f"    MAE: {mae_cn:.2f} years")
    print(f"    R²:  {r2_cn:.3f}")
    print(f"    Pearson r:   {r_pearson:.3f} (p={p_pearson:.3e})")
    
    return {
        'cn_pred_corr': cn_preds_corr,
        'ad_pred_corr': ad_preds_corr,
        'mae_cn': mae_cn,
        'r_pearson': r_pearson
    }

if __name__ == "__main__":
    df, X_ll, X_rr, X_lr = load_data()
    
    valid_mask = df['dx'].isin(['CN', 'AD']).values
    df = df[valid_mask].reset_index(drop=True)
    X_ll = X_ll[valid_mask]
    X_rr = X_rr[valid_mask]
    X_lr = X_lr[valid_mask]
    
    X_full = np.hstack([X_ll, X_rr, X_lr])
    
    cn_mask = (df['dx'] == 'CN').values
    ad_mask = (df['dx'] == 'AD').values
    
    df_cn = df[cn_mask].reset_index(drop=True)
    df_ad = df[ad_mask].reset_index(drop=True)
    
    print(f"\nTotal healthy controls (CN): {len(df_cn)}")
    print(f"Total Alzheimer's patients (AD): {len(df_ad)}")
    
    models = [
        ("Full Matrix (Concat)", X_full),
        ("Left Intra (LL)", X_ll),
        ("Right Intra (RR)", X_rr),
        ("Inter (LR)", X_lr)
    ]
    
    results = {}
    
    for name, data in models:
        X_cn = data[cn_mask]
        X_ad = data[ad_mask]
        y_cn = df_cn['chronological_age'].values
        y_ad = df_ad['chronological_age'].values
        
        res = train_normative_and_predict(X_cn, y_cn, X_ad, name)
        results[name] = res
        
    # Brain Age Prediction Plot
    plt.figure(figsize=(20, 5))
    for i, (name, res) in enumerate(results.items()):
        plt.subplot(1, 4, i+1)
        sns.regplot(x=df_cn['chronological_age'], y=res['cn_pred_corr'], label='CN Controls', color='teal', scatter_kws={'alpha':0.6})
        plt.scatter(df_ad['chronological_age'], res['ad_pred_corr'], color='crimson', label='AD Patients', alpha=0.8, edgecolors='black', s=80)
        
        # Identity line
        all_ages = np.concatenate([df_cn['chronological_age'].values, df_ad['chronological_age'].values])
        all_preds = np.concatenate([res['cn_pred_corr'], res['ad_pred_corr']])
        min_val = min(all_ages.min(), all_preds.min()) - 2
        max_val = max(all_ages.max(), all_preds.max()) + 2
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identity Line')
        
        plt.title(f"{name}\nCN MAE: {res['mae_cn']:.2f} yrs | r: {res['r_pearson']:.2f}")
        plt.xlabel("Chronological Age")
        plt.ylabel("Predicted Brain Age")
        plt.legend(loc="upper left")
        
    plt.tight_layout()
    pred_plot_path = os.path.join(OUTPUT_DIR, "ridge_brain_age_predictions.png")
    plt.savefig(pred_plot_path, dpi=300)
    print(f"\nSaved age prediction plot to: {pred_plot_path}")
    print("Done!")
