import pandas as pd
from datetime import datetime
import json
import os

def get_memberships():
    """
    Pull membership data from Capitan API.
    Returns the raw membership data.
    """
    # TODO: Implement actual API call to Capitan
    # For now, we'll load from JSON file as in the original code
    with open('data/raw_data/capitan_customer_memberships.json', 'r') as f:
        membership_data = json.load(f)
    return membership_data

def process_membership_data(membership_data):
    """
    Process raw membership data into a format that can be used to show membership for a given day.
    Returns a DataFrame with processed membership data.
    """
    membership_data_list = []
    
    # Process each membership
    for membership in membership_data.get('results', []):
        # Get frequency from interval and name
        interval = membership.get('interval', '').upper()
        name = str(membership.get('name', '')).lower()
        
        # Initialize all category flags
        is_founder = 'founder' in name
        is_college = 'college' in name
        is_corporate = 'corporate' in name or 'tfnb' in name or 'founders business' in name
        is_mid_day = 'mid-day' in name or 'mid day' in name
        is_fitness_only = 'fitness only' in name or 'fitness-only' in name
        has_fitness_addon = 'fitness' in name and not is_fitness_only
        is_team_dues = 'team dues' in name or 'team-dues' in name
        is_bcf = 'bcf' in name or 'staff' in name
        
        # Determine size
        if 'family' in name:
            size = 'family'
        elif 'duo' in name:
            size = 'duo'
        elif 'corporate' in name or 'tfnb' in name or 'founders business' in name:
            size = 'corporate'
        else:
            size = 'solo'  # Default to solo if not specified
        
        # Determine frequency
        if '3 month' in name or '3-month' in name:
            frequency = 'prepaid_3mo'
        elif '6 month' in name or '6-month' in name:
            frequency = 'prepaid_6mo'
        elif '12 month' in name or '12-month' in name:
            frequency = 'prepaid_12mo'
        elif is_mid_day:
            frequency = 'bi_weekly'
        elif is_bcf:
            frequency = 'bi_weekly'
        elif interval == 'BWK':
            frequency = 'bi_weekly'
        elif interval == 'MON':
            frequency = 'monthly'
        elif interval == 'YRL' or interval == 'YEA':
            frequency = 'annual'
        elif interval == '3MO':
            frequency = 'prepaid_3mo'
        elif interval == '6MO':
            frequency = 'prepaid_6mo'
        elif interval == '12MO':
            frequency = 'prepaid_12mo'
        else:
            frequency = 'unknown'
        
        # Get start and end dates
        start_date = pd.to_datetime(membership.get('start_date'), errors='coerce')
        end_date = pd.to_datetime(membership.get('end_date'), errors='coerce')
        
        # Skip memberships with invalid dates
        if pd.isna(start_date) or pd.isna(end_date):
            continue
        
        membership_data_list.append({
            'customer_id': membership.get('customer_id'),
            'name': membership.get('name', ''),
            'frequency': frequency,
            'size': size,
            'is_founder': is_founder,
            'is_college': is_college,
            'is_corporate': is_corporate,
            'is_mid_day': is_mid_day,
            'is_fitness_only': is_fitness_only,
            'has_fitness_addon': has_fitness_addon,
            'is_team_dues': is_team_dues,
            'start_date': start_date,
            'end_date': end_date,
            'billing_amount': membership.get('billing_amount'),
            'interval': interval,
            'status': membership.get('status', '')
        })
    
    return pd.DataFrame(membership_data_list)

def get_active_memberships_for_date(df, target_date):
    """
    Get all active memberships for a specific date.
    
    Args:
        df: DataFrame with processed membership data
        target_date: datetime object for the target date
    
    Returns:
        DataFrame with only the memberships active on the target date
    """
    return df[
        (df['start_date'] <= target_date) & 
        (df['end_date'] >= target_date)
    ]

def get_membership_counts_by_frequency(df, target_date):
    """
    Get counts of active memberships by frequency for a specific date.
    
    Args:
        df: DataFrame with processed membership data
        target_date: datetime object for the target date
    
    Returns:
        Dictionary with frequency counts
    """
    active_memberships = get_active_memberships_for_date(df, target_date)
    return active_memberships['frequency'].value_counts().to_dict()

def get_membership_counts_by_size(df, target_date):
    """
    Get counts of active memberships by size for a specific date.
    
    Args:
        df: DataFrame with processed membership data
        target_date: datetime object for the target date
    
    Returns:
        Dictionary with size counts
    """
    active_memberships = get_active_memberships_for_date(df, target_date)
    return active_memberships['size'].value_counts().to_dict()

def get_membership_counts_by_category(df, target_date):
    """
    Get counts of active memberships by category for a specific date.
    
    Args:
        df: DataFrame with processed membership data
        target_date: datetime object for the target date
    
    Returns:
        Dictionary with category counts
    """
    active_memberships = get_active_memberships_for_date(df, target_date)
    
    categories = {
        'founder': active_memberships['is_founder'].sum(),
        'college': active_memberships['is_college'].sum(),
        'corporate': active_memberships['is_corporate'].sum(),
        'mid_day': active_memberships['is_mid_day'].sum(),
        'fitness_only': active_memberships['is_fitness_only'].sum(),
        'has_fitness_addon': active_memberships['has_fitness_addon'].sum(),
        'team_dues': active_memberships['is_team_dues'].sum()
    }
    
    return categories

def export_membership_data(df, detailed=False):
    """
    Export membership data to CSV files.
    
    Args:
        df: DataFrame with processed membership data
        detailed: If True, includes additional customer information
    """
    # Create cache directory if it doesn't exist
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Export basic membership data
    df.to_csv(f'{cache_dir}/current_memberships.csv', index=False)
    print(f"Exported {len(df)} memberships to {cache_dir}/current_memberships.csv")
    
    if detailed:
        # Export detailed membership data with additional customer information
        detailed_df = df.copy()
        detailed_df.to_csv(f'{cache_dir}/detailed_memberships.csv', index=False)
        print(f"Exported {len(detailed_df)} detailed memberships to {cache_dir}/detailed_memberships.csv") 