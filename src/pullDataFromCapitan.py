## Import everything; makes sure you have everything in your python environment
import pandas as pd
import requests
import json

## JUST FOR LOCAL
import os 

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
    'annual': 'Annual',
    'weekly': 'weekly',
    'monthly': 'Monthly',
    'founders': 'monthly' # founders charged monthly
}
bcf_fam_friend_keywords = {
    'bcf family': True,
    'bcf staff': True,
}

## Get results from responses
def get_results_from_api(url, headers):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print("Successful response from " + url)
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
    json_data = response.json()
    results = json_data['results']
    df = pd.DataFrame(results)
    return df

# Define a function to categorize transactions and membership types
def categorize_transaction(description):
    description = description.lower()  # Make it case-insensitive
    
    # Default values
    category = 'Retail'
    membership_size = None
    membership_freq = None
    is_founder = False
    is_bcf_staff_or_friend = False
    
    # Categorize transaction
    for keyword, cat in revenue_category_keywords.items():
        if keyword in description:
            category = cat
            break
    
    # Categorize membership type (only if it's a membership-related transaction)
    for keyword, mem_size in membership_size_keywords.items():
        if keyword in description:
            membership_size = mem_size
            break
            
    # Categorize membership frequency (only if it's a membership-related transaction)
    for keyword, mem_freq in membership_frequency_keywords.items():
        if keyword in description:
            membership_freq = mem_freq
            break

    if 'founder' in description:
        is_founder = True

    if 'bcf family' in description or 'bcf staff' in description:
        is_bcf_staff_or_friend = True
    
    return category, membership_size, membership_freq, is_founder, is_bcf_staff_or_friend

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
        df['invoice_description'].apply(lambda x: pd.Series(categorize_transaction(x)))

    # Convert the 'created_at' column to datetime and extract the date
    df['created_at'] = pd.to_datetime(df['created_at'])  # Convert to datetime
    df['date'] = df['created_at'].dt.date  # Extract just the date (no time)

    # Convert the amounts columns to numeric values (handles strings and errors)
    df['amount_pre_tax'] = pd.to_numeric(df['amount_pre_tax'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['tax_amount'] = pd.to_numeric(df['tax_amount'], errors='coerce')
    df['discount_amount'] = pd.to_numeric(df['discount_amount'], errors='coerce')
    
    return df

def save_data(df, file_name):
    df.to_csv('../data/outputs/' + file_name + '.csv', index=False)
    print(file_name + ' saved in ' + '/data/outputs/')

if __name__ == "__main__":
    df = get_results_from_api(url_payments, headers)
    df = transform_payments_data(df)
    save_data(df, 'payment_data')
