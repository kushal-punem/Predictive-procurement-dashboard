"""
ETL pipeline for the Predictive Procurement Analytics project.

This module ingests real augmented data from the university SIS/ERP,
cleans/normalizes the datasets, and prepares feature tables for the ML models
and dashboard.
"""

from __future__ import annotations

import glob
import os
import pandas as pd
import numpy as np
import pickle

def load_master_data(data_dir: str = "new/Cleaned", load_all: bool = True, save_to_csv: bool = True, output_file: str = "master_data.csv", compute_features: bool = True, use_cache: bool = True) -> pd.DataFrame:
    """
    Load all CSV files from the Cleaned folder and create a single master dataframe.
    
    Parameters:
    -----------
    data_dir : str
        Directory path containing the CSV files (default: "new/Cleaned")
    load_all : bool
        If True, load all data. If False, load a sample (default: True)
    save_to_csv : bool
        If True, save master dataframe to CSV file (default: True)
    output_file : str
        Path/filename for saving the master dataframe (default: "master_data.csv")
    compute_features : bool
        If True, compute feature engineering columns (default: True)
    use_cache : bool
        If True, use cached pickle file if available (default: True)
    
    Returns:
    --------
    pd.DataFrame
        Master dataframe containing all data from all CSV files
    """
    cache_file = "master_data_features.pkl"
    
    # Try to load from cache if requested and available
    if use_cache and compute_features and os.path.exists(cache_file):
        try:
            print(f"Loading cached feature data from {cache_file}...")
            with open(cache_file, 'rb') as f:
                master_df = pickle.load(f)
            print(f"✓ Loaded cached data: {len(master_df)} rows with {len(master_df.columns)} columns")
            return master_df
        except Exception as e:
            print(f"Warning: Could not load cache: {e}. Recomputing features...")
    
    file_pattern = os.path.join(data_dir, "*_frac.csv")
    csv_files = sorted(glob.glob(file_pattern))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir} matching pattern {file_pattern}")
    
    print(f"Found {len(csv_files)} CSV files to load...")
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, engine='c')
            # Extract college name from filename
            # Format: Student_Book_Interactions_COLLEGE_frac.csv
            filename = os.path.basename(f)
            college_name = filename.replace("Student_Book_Interactions_", "").replace("_frac.csv", "")
            df["College"] = college_name
            print(f"  Loaded: {os.path.basename(f)} ({len(df)} rows) - College: {college_name}")
            dfs.append(df)
        except Exception as e:
            print(f"  Error reading {f}: {e}")
    
    if not dfs:
        raise ValueError("No CSV files could be loaded successfully")
    
    # Concatenate all dataframes into one master dataframe
    master_df = pd.concat(dfs, ignore_index=True)
    print(f"\nMaster dataframe created with {len(master_df)} total rows and {len(master_df.columns)} columns")
    
    # Add filter columns needed for dashboard
    master_df["Year"] = master_df["term_year"].astype(str)
    master_df["Semester"] = master_df["term_code"].fillna("Unknown").astype(str)
    
    # Extract department from section_id
    def extract_dept(sec):
        parts = str(sec).split("-")
        return parts[3] if len(parts) > 3 else "GEN"
    master_df["Department"] = master_df["section_id"].apply(extract_dept)
    
    # Map student status
    student_type_map = {"F": "Full-Time", "P": "Part-Time", "H": "Half-Time", "L": "Unknown"}
    master_df["Student_Status"] = master_df["student_full_part_time_status"].map(student_type_map).fillna("Full-Time")
    
    # Compute feature engineering columns if requested
    if compute_features:
        # Pricing & Economic Features
        retail_new = pd.to_numeric(master_df["retail_new"], errors='coerce').fillna(100.0)
        retail_rent = pd.to_numeric(master_df["retail_new_rent"], errors='coerce').fillna(50.0)
        
        retail_new_safe = retail_new.replace(0.0, 100.0)
        ratio = retail_rent / retail_new_safe
        master_df["Rental_to_Retail_Ratio"] = ratio.clip(0.0, 1.5)
        master_df["Arbitrage_Index"] = 1.0 - master_df["Rental_to_Retail_Ratio"]
        
        # Wallet pressure
        afford_score = pd.to_numeric(master_df["price_affordability_score"], errors='coerce').fillna(300.0)
        max_score = afford_score.max() if afford_score.max() > 0 else 1.0
        master_df["Wallet_Pressure_Score"] = (afford_score / max_score).clip(0.0, 1.0)
        
        master_df["Digital_Lock_Flag"] = master_df["ebook_ind"].fillna(0.0)
        
        # Raw features for ML
        master_df["family_annual_income"] = pd.to_numeric(master_df["family_annual_income"], errors='coerce').fillna(40000)
        master_df["has_scholarship"] = pd.to_numeric(master_df["has_scholarship"], errors='coerce').fillna(0)
        master_df["has_loan"] = pd.to_numeric(master_df["has_loan"], errors='coerce').fillna(0)
        master_df["is_rental"] = pd.to_numeric(master_df["is_rental"], errors='coerce').fillna(0)
        
        # Target variable
        master_df["Actual_Purchase_Flag"] = pd.to_numeric(master_df["will_buy"], errors='coerce').fillna(1)
    
    # Save to pickle cache if features were computed
    if compute_features and use_cache:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(master_df, f)
            cache_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
            print(f"✓ Cached to: {cache_file} ({cache_size_mb:.2f} MB)")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    # Save to CSV if requested
    if save_to_csv:
        try:
            master_df.to_csv(output_file, index=False)
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"Master dataframe saved to: {output_file} ({file_size_mb:.2f} MB)")
        except Exception as e:
            print(f"Warning: Could not save master dataframe to {output_file}: {e}")
    
    return master_df


def load_feature_table(data_dir: str = "new/Cleaned", sample_frac: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """
    Load and map the augmented real dataset to the dashboard schema.
    Since the dataset is very large, sample_frac allows loading a subset to keep the dashboard responsive.
    """
    file_pattern = os.path.join(data_dir, "*_frac.csv")
    csv_files = glob.glob(file_pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching {file_pattern}")
        
    dfs = []
    # Use a fixed random generator
    rng = np.random.default_rng(seed)
    
    for f in csv_files:
        try:
            # We sample a fraction to keep memory footprint low
            df = pd.read_csv(f, engine='c')
            # Extract college name from filename
            filename = os.path.basename(f)
            college_name = filename.replace("Student_Book_Interactions_", "").replace("_frac.csv", "")
            df["College"] = college_name
            if sample_frac < 1.0:
                df = df.sample(frac=sample_frac, random_state=seed)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Map physical columns to what the dashboard and model expect
    mapped_df = pd.DataFrame()
    
    # College information
    mapped_df["College"] = df["College"]
    
    # Term information - keep separate for filtering
    mapped_df["Year"] = df["term_year"].astype(str)
    mapped_df["Semester"] = df["term_code"].fillna("Unknown").astype(str)
    mapped_df["Term"] = df["term_code"].fillna("Unknown") + " " + df["term_year"].astype(str)
    
    # Department information
    def extract_dept(sec):
        parts = str(sec).split("-")
        return parts[3] if len(parts) > 3 else "GEN"
    mapped_df["Department"] = df["section_id"].apply(extract_dept)
    mapped_df["Dept_Code"] = mapped_df["Department"]
    
    mapped_df["Title"] = df["title"].fillna("Unknown Title").astype(str)
    mapped_df["Publisher"] = df["author"].fillna("Unknown Author").astype(str).str.slice(0, 20)
    
    student_type_map = {"F": "Full-Time", "P": "Part-Time", "H": "Half-Time", "L": "Unknown"}
    mapped_df["Student_Type"] = df["student_full_part_time_status"].map(student_type_map).fillna("Full-Time")
    mapped_df["Student_Status"] = mapped_df["Student_Type"]  # Alias for filtering
    
    mapped_df["Format"] = df["ebook_ind"].apply(lambda x: "Digital" if x == 1.0 else "Physical")
    
    # Pricing & Economic Features
    retail_new = pd.to_numeric(df["retail_new"], errors='coerce').fillna(100.0)
    retail_rent = pd.to_numeric(df["retail_new_rent"], errors='coerce').fillna(50.0)
    
    retail_new_safe = retail_new.replace(0.0, 100.0)
    ratio = retail_rent / retail_new_safe
    mapped_df["Rental_to_Retail_Ratio"] = ratio.clip(0.0, 1.5)
    
    # Avoid zero division Arbitrage_Index
    mapped_df["Arbitrage_Index"] = 1.0 - mapped_df["Rental_to_Retail_Ratio"]
    
    # Wallet pressure
    afford_score = pd.to_numeric(df["price_affordability_score"], errors='coerce').fillna(300.0)
    max_score = afford_score.max() if afford_score.max() > 0 else 1.0
    mapped_df["Wallet_Pressure_Score"] = (afford_score / max_score).clip(0.0, 1.0)
    
    mapped_df["Digital_Lock_Flag"] = df["ebook_ind"].fillna(0.0)
    
    # Synthetic proxies
    mapped_df["Major_Alignment_Score"] = rng.uniform(0.5, 1.0, size=len(mapped_df))
    mapped_df["Commuter_Friction"] = rng.uniform(0.1, 0.9, size=len(mapped_df))
    
    # Labels
    will_buy = pd.to_numeric(df["will_buy"], errors='coerce').fillna(1)
    mapped_df["Actual_Purchase_Flag"] = will_buy
    mapped_df["Opt_Out_Probability"] = 1.0 - will_buy
    
    mapped_df["Predicted_Demand_Units"] = 1
    mapped_df["Unit_Price"] = retail_new.clip(0.01) # Avoid zero price entirely
    
    # By default set pred to actual, ML model replaces this
    mapped_df["Predicted_Purchase_Prob"] = will_buy
    mapped_df["Projected_Spend"] = mapped_df["Predicted_Demand_Units"] * mapped_df["Unit_Price"] * mapped_df["Predicted_Purchase_Prob"]
    
    # Raw features for ML
    mapped_df["family_annual_income"] = pd.to_numeric(df["family_annual_income"], errors='coerce').fillna(40000)
    mapped_df["has_scholarship"] = pd.to_numeric(df["has_scholarship"], errors='coerce').fillna(0)
    mapped_df["has_loan"] = pd.to_numeric(df["has_loan"], errors='coerce').fillna(0)
    mapped_df["is_rental"] = pd.to_numeric(df["is_rental"], errors='coerce').fillna(0)
    
    return mapped_df

__all__ = ["load_master_data", "load_feature_table"]
