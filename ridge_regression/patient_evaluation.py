import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "/Users/rushilgargash/Desktop/mlpr/numpy_extracted"
ADNIMERGE_PATH = "/Users/rushilgargash/Desktop/mlpr/ADNIMERGE_23Apr2026.csv"

# 11 Patients (9 AD + 2 LMCI) from the unzipped dataset
PATIENT_IDS = [
    "016S6839", "022S6013", "022S6796", "023S6661", "057S6869",
    "126S4507", "126S6721", "127S6433", "137S4862", "168S6735", "168S6754"
]

def load_demographics():
    print("Loading ADNIMERGE to extract demographics...")
    df = pd.read_csv(ADNIMERGE_PATH, usecols=['PTID', 'VISCODE', 'AGE', 'DX_bl'])
    df_bl = df[df['VISCODE'] == 'bl'].drop_duplicates(subset=['PTID'])
    df_bl['clean_id'] = df_bl['PTID'].str.replace('_', '')
    return df_bl

def load_subject_features(sub_id, df_bl):
    paths = [os.path.join(DATA_DIR, f"{sub_id}_{h}.npy") for h in ['ll', 'rr', 'lr']]
    if not all(os.path.exists(p) for p in paths):
        return None
    
    sub_meta = df_bl[df_bl['clean_id'] == sub_id]
    if len(sub_meta) == 0:
        return None
        
    age = sub_meta.iloc[0]['AGE']
    dx = sub_meta.iloc[0]['DX_bl']
    
    if pd.isna(age):
        return None
        
    ll = np.load(paths[0])
    rr = np.load(paths[1])
    lr = np.load(paths[2])
    
    # Fisher z-transform
    ll_z = np.arctanh(np.clip(ll, -0.9999, 0.9999))
    rr_z = np.arctanh(np.clip(rr, -0.9999, 0.9999))
    lr_z = np.arctanh(np.clip(lr, -0.9999, 0.9999))
    
    return {
        'id': sub_id,
        'age': age,
        'dx': dx,
        'll': ll_z,
        'rr': rr_z,
        'lr': lr_z
    }

def main():
    df_bl = load_demographics()
    
    files = os.listdir(DATA_DIR)
    all_subject_ids = sorted(list(set([f.split('_')[0] for f in files if f.endswith('.npy')])))
    
    cn_subjects = []
    new_test_subjects = []
    
    for sub_id in all_subject_ids:
        subj_data = load_subject_features(sub_id, df_bl)
        if subj_data is None:
            continue
            
        dx = subj_data['dx']
        if dx == 'CN':
            cn_subjects.append(subj_data)
        else:
            new_test_subjects.append(subj_data)
            
    print(f"Loaded {len(cn_subjects)} Healthy Controls (CN)")
    print(f"Loaded {len(new_test_subjects)} Test Patients/Subjects")
    
    def get_arrays(subj_list):
        ll = np.array([s['ll'] for s in subj_list])
        rr = np.array([s['rr'] for s in subj_list])
        lr = np.array([s['lr'] for s in subj_list])
        full = np.hstack([ll, rr, lr])
        ages = np.array([s['age'] for s in subj_list])
        return ll, rr, lr, full, ages
        
    X_cn_ll, X_cn_rr, X_cn_lr, X_cn_full, y_cn = get_arrays(cn_subjects)
    X_test_ll, X_test_rr, X_test_lr, X_test_full, y_test = get_arrays(new_test_subjects)
    
    models = {
        'left': (X_cn_ll, X_test_ll),
        'right': (X_cn_rr, X_test_rr),
        'inter': (X_cn_lr, X_test_lr),
        'full': (X_cn_full, X_test_full)
    }
    
    for model_name, (X_cn_m, X_test_m) in models.items():
        selector = VarianceThreshold(threshold=0.0001)
        X_cn_sel = selector.fit_transform(X_cn_m)
        X_test_sel = selector.transform(X_test_m)
        
        scaler = StandardScaler()
        X_cn_scaled = scaler.fit_transform(X_cn_sel)
        X_test_scaled = scaler.transform(X_test_sel)
        
        pca = PCA(n_components=0.90, svd_solver='full', random_state=42)
        X_cn_pca = pca.fit_transform(X_cn_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        ridge = RidgeCV(alphas=np.logspace(-1.5, 1.5, 40))
        ridge.fit(X_cn_pca, y_cn)
        
        test_preds = ridge.predict(X_test_pca)
        
        for idx, s in enumerate(new_test_subjects):
            s[f'pred_age_{model_name}'] = test_preds[idx]
            s[f'bag_{model_name}'] = test_preds[idx] - s['age']
            
    # Filter and display patient evaluation results
    patients = [s for s in new_test_subjects if s['id'] in PATIENT_IDS]
    patients = sorted(patients, key=lambda x: (x['dx'] != 'AD', x['id']))
    
    print("\n11 Clinical Patients Evaluation (Ridge Predictions)")
    print(f"{'Patient ID':<12} | {'Diagnosis':<9} | {'Actual Age':<10} | {'Pred Age (Global)':<17} | {'BAG Left':<10} | {'BAG Right':<10} | {'BAG Inter':<10} | {'BAG Global':<10}")
    for p in patients:
        print(f"{p['id']:<12} | {p['dx']:<9} | {p['age']:>8.1f} y | {p['pred_age_full']:>15.1f} y | {p['bag_left']:>8.2f} y | {p['bag_right']:>8.2f} y | {p['bag_inter']:>8.2f} y | {p['bag_full']:>8.2f} y")

if __name__ == "__main__":
    main()
