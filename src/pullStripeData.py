import stripe
import os
import datetime
import pandas as pd

class pullStripeData:
    ## Dictionaries for processing string in decripitions
    revenue_category_keywords = {
        'day pass': 'Day Pass',
        'team dues': 'Team', 
        'entry pass': 'Day Pass',
        'initial payment': 'New Membership',
        'renewal payment': 'Membership Renewal',
        'fitness':'programming',
        'transformation':'programming',
        'climbing technique':'programming',
        'comp':'programming',
        'class':'programming',
        'booking': 'Event Booking',
        'event': 'Event Booking',
        'birthday': 'Event Booking',
        'membership': 'Membership Renewal',
        'reservation': 'Event Booking'
        # 'capitan': 'Day Pass', ## Just for Square
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

    def save_data(df, file_name):
        df.to_csv('data/outputs/' + file_name + '.csv', index=False)
        print(file_name + ' saved in ' + '/data/outputs/')


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
        for keyword, cat in pullStripeData.revenue_category_keywords.items():
            if keyword in description:
                category = cat
                break
        
        # Categorize membership type (only if it's a membership-related transaction)
        for keyword, mem_size in pullStripeData.membership_size_keywords.items():
            if keyword in description:
                membership_size = mem_size
                break
                
        # Categorize membership frequency (only if it's a membership-related transaction)
        for keyword, mem_freq in pullStripeData.membership_frequency_keywords.items():
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
            df['Description'].apply(lambda x: pd.Series(pullStripeData.categorize_transaction(x)))

        # Convert 'Date' to datetime and handle different formats
        df['date_'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)

        # Extract just the date (without time)
        df['Date'] = df['date_'].dt.date

        # Convert the amounts columns to numeric values (handles strings and errors)
        # df['Total Amount'] = pd.to_numeric(df['Total Amount'], errors='coerce')
        df['Tax Amount'] = pd.to_numeric(df['Tax Amount'], errors='coerce')
        df['Pre-Tax Amount'] = pd.to_numeric(df['Pre-Tax Amount'], errors='coerce')
        df['Data Source'] = 'Stripe'
        
        return df

    def get_balance_transaction_fees(charge):
        balance_transaction_id = charge.get('balance_transaction')
        if balance_transaction_id:
            balance_transaction = stripe.BalanceTransaction.retrieve(balance_transaction_id)
            fee_details = balance_transaction.get('fee_details', [])
            # Extract tax/fee amounts if available
            for fee in fee_details:
                if fee.get('type') == 'tax':
                    return fee.get('amount', 0) / 100  # Tax amount in dollars
        return 0

    def pull_stripe_payments_data_raw(stripe_key, start_date, end_date):
        stripe.api_key = stripe_key
        charges = stripe.Charge.list(
            created={
                'gte': int(start_date.timestamp()),  # Start date in Unix timestamp
                'lte': int(end_date.timestamp())     # End date in Unix timestamp
            },
            limit=100
        )
        
        data = []
        for charge in charges.auto_paging_iter():  # Use pagination for large data
                created_at = datetime.datetime.fromtimestamp(charge['created'])  # Convert from Unix timestamp
                total_money = charge['amount'] / 100  # Stripe amounts are in cents
                pre_tax_money = total_money  # Stripe does not separate pre-tax amount by default
                tax_money = 0 ## takes way too long ## get_balance_transaction_fees(charge)  # Tax amount
                discount_money = charge.get('discount', {}).get('amount', 0) / 100  # Discount amount if available
                currency = charge['currency']
                description = charge.get('description', 'No Description')
                name = charge.get('billing_details', {}).get('name', 'No Name')
                
                data.append({
                    'Description': description,
                    'Pre-Tax Amount': pre_tax_money,
                    'Tax Amount': tax_money,
                    'Discount Amount': discount_money,
                    'Name': name,
                    'Date': created_at.date(),
                })

        # Create DataFrame
        df = pd.DataFrame(data)
        return df

    def pull_and_transform_stripe_payment_data(start_date, end_date):
        # Get your Square Access Token from environment variables
        # Set the API key for authentication
        stripe_key = os.getenv('STRIPE_PRODUCTION_API_KEY')

        df = pullStripeData.pull_stripe_payments_data_raw(stripe_key, start_date, end_date)
        df = pullStripeData.transform_payments_data(df)
        pullStripeData.save_data(df, 'stripe_transaction_data')
        return df


if __name__ == "__main__":
    # Get today's date and calculate the start date for the last year
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)
    pull_stripe = pullStripeData()
    pull_stripe.pull_and_transform_stripe_payment_data(start_date, end_date)
