import pandas as pd
from datetime import timedelta

def calculate_membership_metrics(df):
    # Filter out rows with amount <= 1 (invalid transactions)
    df = df[df['amount'] > 1]

    print(f"df is len {len(df)}")

    # Convert 'created_at' to datetime and make sure it is timezone-naive
    # df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)
    df.loc[:, 'created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)

    # Create a list of dates from August 1 to today
    start_date = pd.to_datetime('2024-08-01')
    end_date = pd.to_datetime('today')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a dictionary to store results
    results = []

    # Loop over each date in the date range
    for date in date_range:
        # Ensure date is also timezone-naive to match the 'created_at' column
        date = date.tz_localize(None)

        # Get the past 370 days for yearly memberships, 40 days for monthly, and 21 days for weekly
        df_yearly = df[(df['created_at'] >= date - timedelta(days=370)) & (df['created_at'] <= date)]
        df_monthly = df[(df['created_at'] >= date - timedelta(days=40)) & (df['created_at'] <= date)]
        df_weekly = df[(df['created_at'] >= date - timedelta(days=21)) & (df['created_at'] <= date)]
        print(f"Date: {date}")
        print(f"Yearly memberships count: {len(df_yearly)}")
        print(f"Monthly memberships count: {len(df_monthly)}")
        print(f"Weekly memberships count: {len(df_weekly)}")
        # print(df_weekly.head(3))

        # Filter yearly solo memberships
        yearly_solo = df_yearly[(df_yearly['membership_freq'] == 'yearly') & (df_yearly['membership_size'] == 'Solo')]

        # Filter yearly duo memberships
        yearly_duo = df_yearly[(df_yearly['membership_freq'] == 'yearly') & (df_yearly['membership_size'] == 'Duo')]

        # Filter yearly family memberships
        yearly_family = df_yearly[(df_yearly['membership_freq'] == 'yearly') & (df_yearly['membership_size'] == 'Family')]

        # Now repeat the same for monthly and weekly memberships

        # Monthly solo memberships
        monthly_solo = df_monthly[(df_monthly['membership_freq'] == 'monthly') & (df_monthly['membership_size'] == 'Solo')]

        # Monthly duo memberships
        monthly_duo = df_monthly[(df_monthly['membership_freq'] == 'monthly') & (df_monthly['membership_size'] == 'Duo')]

        # Monthly family memberships
        monthly_family = df_monthly[(df_monthly['membership_freq'] == 'monthly') & (df_monthly['membership_size'] == 'Family')]

        # Weekly solo memberships
        weekly_solo = df_weekly[(df_weekly['membership_freq'] == 'weekly') & (df_weekly['membership_size'] == 'Solo')]

        # Weekly duo memberships
        weekly_duo = df_weekly[(df_weekly['membership_freq'] == 'weekly') & (df_weekly['membership_size'] == 'Duo')]

        # Weekly family memberships
        weekly_family = df_weekly[(df_weekly['membership_freq'] == 'weekly') & (df_weekly['membership_size'] == 'Family')]

        # Get the counts of unique emails for each
        yearly_solo_count = yearly_solo['customer_email'].nunique()
        yearly_duo_count = yearly_duo['customer_email'].nunique()
        yearly_family_count = yearly_family['customer_email'].nunique()

        monthly_solo_count = monthly_solo['customer_email'].nunique()
        monthly_duo_count = monthly_duo['customer_email'].nunique()
        monthly_family_count = monthly_family['customer_email'].nunique()

        weekly_solo_count = weekly_solo['customer_email'].nunique()
        weekly_duo_count = weekly_duo['customer_email'].nunique()
        weekly_family_count = weekly_family['customer_email'].nunique()
        print(f"weekly: {weekly_solo_count}")

        # Append the results for this date
        results.append({
            'date': date,
            'yearly_solo': yearly_solo_count,
            'yearly_duo': yearly_duo_count,
            'yearly_family': yearly_family_count,
            'monthly_solo': monthly_solo_count,
            'monthly_duo': monthly_duo_count,
            'monthly_family': monthly_family_count,
            'weekly_solo': weekly_solo_count,
            'weekly_duo': weekly_duo_count,
            'weekly_family': weekly_family_count
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # pullDataFromCapitan.save_data(results_df, 'membership_data.py')

    return results_df

df = pd.read_csv('data/outputs/payment_data.csv')
calculate_membership_metrics(df)