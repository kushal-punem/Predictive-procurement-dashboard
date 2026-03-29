import pandas as pd
from etl_pipeline import load_master_data

def count_student_purchases_fun(term_year='ALL', dept_code='ALL'):
    """
    Calculate purchase statistics from the master dataframe.
    
    Parameters:
    -----------
    term_year : str
        Filter by term year (e.g., '2023' or 'ALL' for all years)
    dept_code : str
        Filter by department code (e.g., '126' or 'ALL' for all departments)
    
    Returns:
    --------
    dict
        Dictionary containing counts: buy_1, buy_0, student_full_time, student_part_time
    """
    # Load all data from Cleaned folder into master dataframe
    try:
        master_df = load_master_data()
    except Exception as e:
        print(f"Error loading master data: {e}")
        return {'buy_1': 0, 'buy_0': 0, 'student_full_time': 0, 'student_part_time': 0}
    
    # Work with a copy to avoid modifying original
    df = master_df.copy()
    
    # Apply term_year filter if specified
    if str(term_year).upper() != 'ALL':
        if 'term_year' in df.columns:
            target_year = str(term_year).strip()[-2:]
            df_years = df['term_year'].astype(str).str.strip().str[-2:]
            df = df[df_years == target_year]
            print(f"Filtered by term_year '{term_year}': {len(df)} rows")
        else:
            print("Notice: 'term_year' column missing in master dataframe. Skipping year filter.")
    
    # Apply dept_code filter if specified
    if str(dept_code).upper() != 'ALL':
        if 'dept_code' in df.columns:
            df = df[df['dept_code'].astype(str).str.strip() == str(dept_code).strip()]
            print(f"Filtered by dept_code '{dept_code}': {len(df)} rows")
        else:
            print("Notice: 'dept_code' column missing in master dataframe. Skipping dept filter.")
    
    # Calculate will_buy counts
    total_will_buy_1 = 0
    total_will_buy_0 = 0
    if 'will_buy' in df.columns:
        counts = df['will_buy'].value_counts()
        total_will_buy_1 = counts.get(1, 0)
        total_will_buy_0 = counts.get(0, 0)
    else:
        print("Notice: 'will_buy' column missing in master dataframe.")
    
    # Calculate student status counts
    student_part_time = 0
    student_full_time = 0
    if 'student_full_part_time_status' in df.columns:
        counts = df['student_full_part_time_status'].value_counts()
        student_full_time = counts.get('F', 0)
        student_part_time = counts.get('P', 0)
    else:
        print("Notice: 'student_full_part_time_status' column missing in master dataframe.")
    
    return {
        'buy_1': total_will_buy_1,
        'buy_0': total_will_buy_0,
        'student_full_time': student_full_time,
        'student_part_time': student_part_time
    }

# Example usage - calculate statistics from all data
funcall = count_student_purchases_fun(term_year='ALL', dept_code='ALL')

print("\n--- Purchase Statistics ---")
print(funcall)