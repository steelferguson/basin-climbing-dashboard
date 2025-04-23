## Import everything; makes sure you have everything in your python environment
import pandas as pd
import requests
import json
from datetime import timedelta
## JUST FOR LOCAL
import os 

class pullDataFromCapitan:
    ## JUST FOR LOCAL, Get the token from the environment variable
    my_token = os.getenv('CAPITAN_API_TOKEN')

    if not my_token:
        raise ValueError("API token not found. Please set CAPITAN_API_TOKEN as an environment variable.")

    ## Make the API call
    ## Set up URLs
    url_base = 'https://api.hellocapitan.com/api/'
    url_payments =  url_base + 'payments/' + '?page=1&page_size=10000000000'
    url_list = [url_payments]

    ## Set up headers
    headers={'Authorization': 'token {}'.format(my_token)}

    ## Dictionaries for processing string in decripitions
    revenue_category_keywords = {
        'day pass': 'Day Pass',
        'team dues': 'Team', 
        'membership renewal': 'Membership Renewal',
        'new membership': 'New Membership',
        'booking': 'Event Booking',
        'event': 'Event Booking'
    }
    membership_size_keywords = {
        'bcf family': 'BCF Staff & Family',
        'bcf staff': 'BCF Staff & Family',
        'duo': 'Duo',
        'solo': 'Solo',
        'family': 'Family',
        'corporate': 'Corporate'
    }
    membership_frequency_keywords = {
        'annual': 'annual',
        'weekly': 'weekly',
        'monthly': 'monthly',
        'founders': 'monthly' # founders charged monthly
    }
    bcf_fam_friend_keywords = {
        'bcf family': True,
        'bcf staff': True,
    }

    @staticmethod
    def save_raw_response(data, filename):
        """Save raw API response to a JSON file."""
        os.makedirs('data/raw_data', exist_ok=True)
        filepath = f'data/raw_data/{filename}.json'
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved raw response to {filepath}")

    def __init__(self):
        self.base_url = 'https://api.hellocapitan.com/api/'
        self.headers = {'Authorization': f'token {self.my_token}'}

    def get_results_from_api(self, url):
        """
        Make API request and handle response.
        """
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                print("Successful response from " + url)
            else:
                print(f"Failed to retrieve data. Status code: {response.status_code}")
            
            json_data = response.json()
            
            # Determine which type of data we're saving based on the URL
            if 'customer-memberships' in url:
                filename = 'capitan_customer_memberships'
            elif 'payments' in url:
                filename = 'capitan_payments'
            else:
                filename = 'capitan_response'
            
            # Save raw response
            self.save_raw_response(json_data, filename)
            
            return json_data
        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            return None

    # Define a function to categorize transactions and membership types
    @staticmethod
    def categorize_transaction(description):
        description = description.lower()  # Make it case-insensitive
        
        # Default values
        category = 'Retail'
        membership_size = None
        membership_freq = None
        is_founder = False
        is_bcf_staff_or_friend = False
        
        # Categorize transaction
        for keyword, cat in pullDataFromCapitan.revenue_category_keywords.items():
            if keyword in description:
                category = cat
                break
        
        # Categorize membership type (only if it's a membership-related transaction)
        for keyword, mem_size in pullDataFromCapitan.membership_size_keywords.items():
            if keyword in description:
                membership_size = mem_size
                break
                
        # Categorize membership frequency (only if it's a membership-related transaction)
        for keyword, mem_freq in pullDataFromCapitan.membership_frequency_keywords.items():
            if keyword in description:
                membership_freq = mem_freq
                break

        if 'founder' in description:
            is_founder = True

        if 'bcf family' in description or 'bcf staff' in description:
            is_bcf_staff_or_friend = True
        
        return category, membership_size, membership_freq, is_founder, is_bcf_staff_or_friend
    
    @staticmethod
    def categorize_revenue_sub_category(description):
        description = description.lower()  # Convert to lowercase for case-insensitive matching
        if "birthday" in description:
            return "birthday party"
        elif "team dues" in description:
            return "team dues"
        elif "comp" in description:
            return "competition"
        elif "climbing technique" in description:
            return "climbing class"
        elif "belay class" in description:
            return "belay class"
        elif "field trip" in description:
            return "private group rental"
        elif "private group" in description:
            return "private group rental"
        elif "transformation" in description:
            return "fitness class"
        elif "fitness" in description:
            return "fitness class"
        else:
            return "other"

    @staticmethod
    def transform_payments_data(df):
        """
        Transforms the payments data by adding new columns and converting data types.

        Parameters:
        df (pd.DataFrame): Original DataFrame to transform
        
        Returns:
        pd.DataFrame: Transformed DataFrame with new columns and type conversions
        """
        # Apply the categorize_transaction function to create new columns
        df[['revenue_category', 'membership_size', 'membership_freq', 'is_founder', 'is_free_membership']] = \
            df['invoice_description'].apply(lambda x: pd.Series(pullDataFromCapitan.categorize_transaction(x)))

        # Convert the 'created_at' column to datetime and extract the date
        # df['created_at'] = pd.to_datetime(df['created_at'])  # Convert to datetime
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert(None)
        df['date'] = df['created_at'].dt.date  # Extract just the date (no time)

        # Convert the amounts columns to numeric values (handles strings and errors)
        df['amount_pre_tax'] = pd.to_numeric(df['amount_pre_tax'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['tax_amount'] = pd.to_numeric(df['tax_amount'], errors='coerce')
        df['discount_amount'] = pd.to_numeric(df['discount_amount'], errors='coerce')

        df['revenue_sub_category'] = df['invoice_description'].apply(pullDataFromCapitan.categorize_revenue_sub_category)
        
        return df

    @staticmethod
    def calculate_membership_metrics(df):
        # Filter out rows with amount <= 1 (invalid transactions)
        df = df[df['amount'] > 1]

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

            # Filter yearly solo memberships
            yearly_solo = df_yearly[(df_yearly['membership_freq'] == 'annual') & (df_yearly['membership_size'] == 'Solo')]

            # Filter yearly duo memberships
            yearly_duo = df_yearly[(df_yearly['membership_freq'] == 'annual') & (df_yearly['membership_size'] == 'Duo')]

            # Filter yearly family memberships
            yearly_family = df_yearly[(df_yearly['membership_freq'] == 'annual') & (df_yearly['membership_size'] == 'Family')]

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
        pullDataFromCapitan.save_data(results_df, 'membership_data.py')

        return results_df
    
    @staticmethod
    def save_data(df, file_name):
        df.to_csv('data/outputs/' + file_name + '.csv', index=False)
        print(file_name + ' saved in ' + '/data/outputs/')

    def pull_and_transform_payment_data(self):
        """
        Pull and transform payment data from Capitan API.
        """
        # Get payments data in a single request with large page size
        response = self.get_results_from_api(self.url_payments)
        
        if not response:
            print("Failed to get payments data from Capitan API")
            return None
            
        if 'results' not in response:
            print("No results found in payments response")
            return None
            
        payments = response['results']
        print(f"Total payments retrieved: {len(payments)}")
        
        # Transform payments data
        df = pd.DataFrame(payments)
        df = self.transform_payments_data(df)
        
        # Save transformed data
        self.save_data(df, 'capitan_data')
        
        return df

    def fetch_and_save_memberships(self):
        """
        Fetch memberships data from Capitan API and save as JSON.
        """
        url = self.url_base + 'customer-memberships/' + '?page=1&page_size=10000000000'
        response = self.get_results_from_api(url)
        
        if not response:
            print("Failed to get memberships from Capitan API")
            return None
            
        if 'results' not in response:
            print("No results found in memberships response")
            return None
            
        memberships = response['results']
        print(f"Total memberships retrieved: {len(memberships)}")
        
        # Save raw response
        self.save_raw_response(response, 'capitan_customer_memberships')
        
        return pd.DataFrame(memberships)

    def get_memberships(self):
        """
        Get memberships from Capitan API.
        """
        # Get memberships data in a single request with large page size
        url = self.url_base + 'customer-memberships/' + '?page=1&page_size=10000000000'
        response = self.get_results_from_api(url)
        
        if not response:
            print("Failed to get memberships from Capitan API")
            return None
            
        if 'results' not in response:
            print("No results found in memberships response")
            return None
            
        memberships = response['results']
        print(f"Total memberships retrieved: {len(memberships)}")
        return pd.DataFrame(memberships)

if __name__ == "__main__":
    pull_capitan = pullDataFromCapitan()
    pull_capitan.pull_and_transform_payment_data()

