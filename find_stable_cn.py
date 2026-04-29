import pandas as pd
import numpy as np

def find_strictly_stable_cn(csv_path):
    """
    Finds subjects who are strictly Cognitively Normal (CN) from baseline 
    throughout all follow-up visits.
    """
    # Load the dataset
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Filter for subjects who started as CN
    # We check the 'DX_bl' column which is the gold standard for baseline diagnosis in ADNI
    df_cn_start = df[df['DX_bl'] == 'CN']
    
    grouped = df_cn_start.groupby('PTID')
    verified_subjects = []
    
    for ptid, group in grouped:
        # 1. Get all unique diagnosis values for this subject from the 'DX' column
        # This column tracks their diagnosis at each specific visit
        all_diagnoses = group['DX'].dropna().unique()
        
        # 2. Check if they ever had anything other than 'CN'
        # If they transition to MCI or Dementia, 'DX' will change.
        # If they remain CN, the only value in all_diagnoses should be 'CN'.
        is_always_cn = len(all_diagnoses) > 0 and all(d == 'CN' for d in all_diagnoses)
        
        # 3. Ensure they have follow-up data
        # We need to see them at least once after baseline ('bl') to confirm they didn't convert
        has_followup = any(v != 'bl' for v in group['VISCODE'].unique())
        
        # FINAL CRITERIA:
        # - Baseline diagnosis was 'CN' (DX_bl == 'CN')
        # - Every subsequent diagnosis recorded was also 'CN'
        # - Has at least one follow-up record to prove 'no Alzheimer's later'
        if is_always_cn and has_followup:
            verified_subjects.append(ptid)
            
    return verified_subjects

if __name__ == "__main__":
    csv_file = "/Users/rushilgargash/Desktop/mlpr/ADNIMERGE_23Apr2026.csv"
    
    print(f"Running strict stability validation on {csv_file}...")
    stable_subjects = find_strictly_stable_cn(csv_file)
    stable_subjects.sort()
    
    print(f"\n--- STRICT VALIDATION RESULTS ---")
    print(f"Total Subjects: {len(stable_subjects)}")
    print("Criteria:")
    print(" 1. Baseline Diagnosis (DX_bl) is strictly 'CN'.")
    print(" 2. All follow-up Diagnosis (DX) entries are strictly 'CN'.")
    print(" 3. Verified follow-up data exists (confirmed stability over time).")
    
    # Save to CSV
    output_file = "stable_cn_subjects.csv"
    output_df = pd.DataFrame(stable_subjects, columns=['PTID'])
    output_df.to_csv(output_file, index=False)
            
    print(f"\nStrictly verified list saved to '{output_file}'.")
    # Final check: Print first 5 to show consistency
    if len(stable_subjects) > 0:
        print("\nVerifying first 5 subjects:")
        df = pd.read_csv(csv_file, low_memory=False)
        for s in stable_subjects[:5]:
            subset = df[df['PTID'] == s][['PTID', 'VISCODE', 'DX', 'DX_bl']].sort_values('VISCODE')
            print(f"\nSubject {s}:")
            print(subset)
