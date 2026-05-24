import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "/Users/rushilgargash/Desktop/mlpr/numpy_extracted"
ADNIMERGE_PATH = "/Users/rushilgargash/Desktop/mlpr/ADNIMERGE_23Apr2026.csv"

# Subject groups definition
ORIGINAL_AD_IDS = ['022S6013', '023S6661', '057S6869', '126S6721', '168S6754']
PATIENT_IDS = ORIGINAL_AD_IDS + ['016S6839', '022S6796', '127S6433', '168S6735', '126S4507', '137S4862']

def load_demographics():
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

def get_arrays(subj_list):
    ll = np.array([s['ll'] for s in subj_list])
    rr = np.array([s['rr'] for s in subj_list])
    lr = np.array([s['lr'] for s in subj_list])
    full = np.hstack([ll, rr, lr])
    ages = np.array([s['age'] for s in subj_list])
    return ll, rr, lr, full, ages

def run_svm_classification():
    print("RUNNING DOWNSTREAM SVM DIAGNOSTIC CLASSIFICATION ")
    
    df_bl = load_demographics()
    files = os.listdir(DATA_DIR)
    all_subject_ids = sorted(list(set([f.split('_')[0] for f in files if f.endswith('.npy')])))
    
    cn_subjects = []
    ad_subjects_original = []
    new_unzipped_subjects = []
    
    for sub_id in all_subject_ids:
        subj_data = load_subject_features(sub_id, df_bl)
        if subj_data is None:
            continue
        dx = subj_data['dx']
        if dx == 'CN' and sub_id not in ['033S4176']:
            cn_subjects.append(subj_data)
        elif sub_id in ORIGINAL_AD_IDS:
            ad_subjects_original.append(subj_data)
        else:
            new_unzipped_subjects.append(subj_data)
            
    print(f"Loaded {len(cn_subjects)} Healthy Controls (CN).")
    print(f"Loaded {len(ad_subjects_original)} Original AD Patients.")
    print(f"Loaded {len(new_unzipped_subjects)} New Unzipped Subjects.")
    
    X_cn_ll, X_cn_rr, X_cn_lr, X_cn_full, y_cn = get_arrays(cn_subjects)
    X_orig_ll, X_orig_rr, X_orig_lr, X_orig_full, y_orig = get_arrays(ad_subjects_original)
    X_new_ll, X_new_rr, X_new_lr, X_new_full, y_new = get_arrays(new_unzipped_subjects)
    
    models = {
        'left': (X_cn_ll, X_orig_ll, X_new_ll),
        'right': (X_cn_rr, X_orig_rr, X_new_rr),
        'inter': (X_cn_lr, X_orig_lr, X_new_lr),
        'full': (X_cn_full, X_orig_full, X_new_full)
    }
    
    cn_bags = {}
    orig_bags = {}
    new_bags = {}
    
    for model_name, (X_cn_m, X_orig_m, X_new_m) in models.items():
        loo = LeaveOneOut()
        cn_preds_raw = np.zeros(len(X_cn_m))
        cn_bags_res = np.zeros(len(y_cn))
        
        for train_idx, val_idx in loo.split(X_cn_m):
            X_tr, X_val = X_cn_m[train_idx], X_cn_m[val_idx]
            y_tr = y_cn[train_idx]
            
            sel_cv = VarianceThreshold(threshold=0.0001)
            X_tr_sel = sel_cv.fit_transform(X_tr)
            X_val_sel = sel_cv.transform(X_val)
            
            scal_cv = StandardScaler()
            X_tr_sc = scal_cv.fit_transform(X_tr_sel)
            X_val_sc = scal_cv.transform(X_val_sel)
            
            pca_cv = PCA(n_components=0.90, svd_solver='full', random_state=42)
            X_tr_pca = pca_cv.fit_transform(X_tr_sc)
            X_val_pca = pca_cv.transform(X_val_sc)
            
            ridge_cv = RidgeCV(alphas=np.logspace(-1.5, 1.5, 40))
            ridge_cv.fit(X_tr_pca, y_tr)
            pred_raw = ridge_cv.predict(X_val_pca)[0]
            
            pred_tr_raw = ridge_cv.predict(X_tr_pca)
            bags_tr_raw = pred_tr_raw - y_tr
            
            lr_bias = LinearRegression()
            lr_bias.fit(y_tr.reshape(-1, 1), bags_tr_raw)
            
            expected_bias = lr_bias.predict(y_cn[val_idx].reshape(-1, 1))[0]
            bag_raw = pred_raw - y_cn[val_idx[0]]
            cn_bags_res[val_idx[0]] = bag_raw - expected_bias
            cn_preds_raw[val_idx[0]] = pred_raw
            
        cn_bags[model_name] = cn_bags_res
        
        selector = VarianceThreshold(threshold=0.0001)
        X_cn_sel = selector.fit_transform(X_cn_m)
        X_orig_sel = selector.transform(X_orig_m)
        X_new_sel = selector.transform(X_new_m)
        
        scaler = StandardScaler()
        X_cn_scaled = scaler.fit_transform(X_cn_sel)
        X_orig_scaled = scaler.transform(X_orig_sel)
        X_new_scaled = scaler.transform(X_new_sel)
        
        pca = PCA(n_components=0.90, svd_solver='full', random_state=42)
        X_cn_pca = pca.fit_transform(X_cn_scaled)
        X_orig_pca = pca.transform(X_orig_scaled)
        X_new_pca = pca.transform(X_new_scaled)
        
        ridge = RidgeCV(alphas=np.logspace(-1.5, 1.5, 40))
        ridge.fit(X_cn_pca, y_cn)
        
        orig_preds_raw = ridge.predict(X_orig_pca)
        orig_bags_raw = orig_preds_raw - y_orig
        
        new_preds_raw = ridge.predict(X_new_pca)
        new_bags_raw = new_preds_raw - y_new
        
        cn_bags_raw_for_final = cn_preds_raw - y_cn
        lr_final = LinearRegression()
        lr_final.fit(y_cn.reshape(-1, 1), cn_bags_raw_for_final)
        
        expected_bias_orig = lr_final.predict(y_orig.reshape(-1, 1))
        orig_bags[model_name] = orig_bags_raw - expected_bias_orig
        
        expected_bias_new = lr_final.predict(y_new.reshape(-1, 1))
        new_bags[model_name] = new_bags_raw - expected_bias_new
        
    for idx, s in enumerate(cn_subjects):
        s['bag_left'] = cn_bags['left'][idx]
        s['bag_right'] = cn_bags['right'][idx]
        s['bag_inter'] = cn_bags['inter'][idx]
        
    for idx, s in enumerate(ad_subjects_original):
        s['bag_left'] = orig_bags['left'][idx]
        s['bag_right'] = orig_bags['right'][idx]
        s['bag_inter'] = orig_bags['inter'][idx]
        
    for idx, s in enumerate(new_unzipped_subjects):
        s['bag_left'] = new_bags['left'][idx]
        s['bag_right'] = new_bags['right'][idx]
        s['bag_inter'] = new_bags['inter'][idx]
        
    main_cohort = cn_subjects + ad_subjects_original
    X_main = np.array([[s['bag_left'], s['bag_right'], s['bag_inter']] for s in main_cohort])
    y_main = np.array([0 if s['dx'] == 'CN' else 1 for s in main_cohort])
    
    loo_svm = LeaveOneOut()
    cv_preds = np.zeros(len(y_main))
    cv_decisions = np.zeros(len(y_main))
    
    for train_idx, val_idx in loo_svm.split(X_main):
        X_tr, X_val = X_main[train_idx], X_main[val_idx]
        y_tr = y_main[train_idx]
        
        svm_cv = SVC(kernel='linear', class_weight='balanced', random_state=42)
        svm_cv.fit(X_tr, y_tr)
        cv_preds[val_idx[0]] = svm_cv.predict(X_val)[0]
        cv_decisions[val_idx[0]] = svm_cv.decision_function(X_val)[0]
        
    acc = accuracy_score(y_main, cv_preds)
    prec = precision_score(y_main, cv_preds, zero_division=0)
    rec = recall_score(y_main, cv_preds, zero_division=0)
    
    print("\nLOOCV Downstream Diagnostic SVM Classification Performance (Main Cohort: N=62)")
    print(f"  * Accuracy:  {acc * 100:.1f}%")
    print(f"  * Precision: {prec * 100:.1f}%")
    print(f"  * Recall:    {rec * 100:.1f}%")
    
    print("\nClassification Report (LOOCV):")
    print(classification_report(y_main, cv_preds, target_names=['CN Controls', 'Alzheimer Patients']))
    
    svm_final = SVC(kernel='linear', class_weight='balanced', random_state=42)
    svm_final.fit(X_main, y_main)
    
    all_subjects = cn_subjects + ad_subjects_original + new_unzipped_subjects
    patients = [s for s in all_subjects if s['id'] in PATIENT_IDS]
    patients = sorted(patients, key=lambda x: (x['dx'] != 'AD', x['id']))
    
    print("\nDiagnostic Classification for all 11 Patients (Test Data Only)")
    print(f"{'Patient ID':<12} | {'Diagnosis':<9} | {'Actual Age':<10} | {'BAG Left':<10} | {'BAG Right':<10} | {'BAG Inter':<10} | {'SVM Pred':<10} | {'SVM Decision':<12} | {'Evaluation Type':<25}")
    
    correct_cnt = 0
    for p in patients:
        X_p = np.array([[p['bag_left'], p['bag_right'], p['bag_inter']]])
        is_loocv = p['id'] in ORIGINAL_AD_IDS
        
        if is_loocv:
            idx = main_cohort.index(p)
            svm_pred = cv_preds[idx]
            decision = cv_decisions[idx]
            eval_type = "Out-of-Fold (LOOCV)"
        else:
            svm_pred = svm_final.predict(X_p)[0]
            decision = svm_final.decision_function(X_p)[0]
            eval_type = "Out-of-Sample (Independent)"
            
        pred_label = "AD" if svm_pred == 1 else "CN"
        if pred_label == "AD":
            correct_cnt += 1
            
        decision_str = f"+{decision:.3f}" if decision >= 0 else f"{decision:.3f}"
        print(f"{p['id']:<12} | {p['dx']:<9} | {p['age']:>8.1f} y | {p['bag_left']:>8.2f} y | {p['bag_right']:>8.2f} y | {p['bag_inter']:>8.2f} y | {pred_label:<10} | {decision_str:<12} | {eval_type:<25}")
        
    print(f"\nSVM Sensitivity (Recall) on the 11 Patients: {correct_cnt/len(patients)*100:.1f}% ({correct_cnt} out of 11 correctly classified)")

if __name__ == "__main__":
    run_svm_classification()
