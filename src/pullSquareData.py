from square.client import Client
from square.http.auth.o_auth_2 import BearerAuthCredentials
import os
import datetime
import pandas as pd

class pullSquareData:
    ## Dictionaries for processing string in decripitions
    revenue_category_keywords = {
        'day pass': 'Day Pass',
        'team dues': 'Team', 
        'membership renewal': 'Membership Renewal',
        'new membership': 'New Membership',
        'fitness':'programming',
        'transformation':'programming',
        'climbing technique':'programming',
        'comp':'programming',
        'class':'programming',
        'event': 'Event Booking',
        'birthday': 'Event Booking',
        'retreat': 'Event Booking',
        'pass': 'Day Pass',
        'booking': 'Event Booking',
        'capitan': 'Day Pass' ## Just for Square
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
        for keyword, cat in pullSquareData.revenue_category_keywords.items():
            if keyword in description:
                category = cat
                break
        
        # Categorize membership type (only if it's a membership-related transaction)
        for keyword, mem_size in pullSquareData.membership_size_keywords.items():
            if keyword in description:
                membership_size = mem_size
                break
                
        # Categorize membership frequency (only if it's a membership-related transaction)
        for keyword, mem_freq in pullSquareData.membership_frequency_keywords.items():
            if keyword in description:
                membership_freq = mem_freq
                break

        if 'founder' in description:
            is_founder = True

        if 'bcf family' in description or 'bcf staff' in description:
            is_bcf_staff_or_friend = True
        
        return category, membership_size, membership_freq, is_founder, is_bcf_staff_or_friend
    
    def count_day_passes(revenue_category, base_amount, total_amount):
        return round(total_amount / base_amount)

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
            df['Description'].apply(lambda x: pd.Series(pullSquareData.categorize_transaction(x)))

        # Convert 'Date' to datetime and handle different formats
        df['date_'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)

        # Extract just the date (without time)
        df['Date'] = df['date_'].dt.date

        # Convert the amounts columns to numeric values (handles strings and errors)
        # df['Total Amount'] = pd.to_numeric(df['Total Amount'], errors='coerce')
        df['Tax Amount'] = pd.to_numeric(df['Tax Amount'], errors='coerce')
        df['Pre-Tax Amount'] = pd.to_numeric(df['Pre-Tax Amount'], errors='coerce')
        df['Pre-Tax Amount'] = df['Pre-Tax Amount'] - df['Discount Amount']
        df['Data Source'] = 'Square'
        
        # Add a column for day pass count using 'Base Price Amount'
        df['Day Pass Count'] = df.apply(
            lambda row: round(row['Total Amount'] / row['base_price_amount'])
            if row['revenue_category'] == 'Day Pass' and row['base_price_amount'] > 0
            else 0,
            axis=1
        )
        
        return df

    def pull_square_payments_data_raw(square_token, location_id, end_time, begin_time, limit):
        # Initialize the Square Client with bearer_auth_credentials
        client = Client(
            bearer_auth_credentials=BearerAuthCredentials(
                access_token=square_token
            ),
            environment='production'
        )
        body = {
            "location_ids": [location_id],
            "query": {
                "filter": {
                    "date_time_filter": {
                        "created_at": {
                            "start_at": begin_time,
                            "end_at": end_time
                        }
                    }
                }
            },
            "limit": limit
        }

        # Fetch all orders using pagination
        orders_list = []
        while True:
            result = client.orders.search_orders(body=body)
            if result.is_success():
                orders = result.body.get('orders', [])
                orders_list.extend(orders)
                cursor = result.body.get('cursor')
                if cursor:
                    body['cursor'] = cursor  # Update body with cursor for next page
                else:
                    break  # Exit loop when no more pages
            elif result.is_error():
                print("Error:", result.errors)
                break

        # Extract relevant data for DataFrame
        data = []
        for order in orders_list:
            created_at = order.get('created_at')  # Order creation date
            line_items = order.get('line_items', [])

            for item in line_items:
                name = item.get('name', 'No Name')
                description = item.get('variation_name', 'No Description')

                # Get the specific amount for each item
                item_total_money = item.get('total_money', {}).get('amount', 0) / 100  # Convert from cents
                _item_pre_tax_money = item.get('base_price_money', {}).get('amount', 0) / 100  # Pre-tax amount (if available)
                item_tax_money = item.get('total_tax_money', {}).get('amount', 0) / 100  # Tax amount for the item
                item_pre_tax_money = item_total_money - item_tax_money
                item_discount_money = item.get('total_discount_money', {}).get('amount', 0) / 100  # Discount for the item

                data.append({
                    'Description': description,
                    'Pre-Tax Amount': item_pre_tax_money,
                    'Tax Amount': item_tax_money,
                    'Discount Amount': item_discount_money,
                    'Name': name,
                    'Total Amount': item_total_money,
                    'Date': created_at,
                    'base_price_amount': _item_pre_tax_money
                })
        
        # Create a DataFrame
        df= pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Drop rows where 'Date' is null
        df = df.dropna(subset=['Date'])
        return df
    
    def pull_square_invoices(square_token, location_id):
        """
        Fetches paid invoices from Square for a specific location, including the date they were paid.

        Parameters:
        square_token (str): The Square API token.
        location_id (str): The location ID for which to fetch invoices.

        Returns:
        pd.DataFrame: A DataFrame containing details of paid invoices.
        """
        # Initialize the Square Client with bearer_auth_credentials
        client = Client(
            bearer_auth_credentials=BearerAuthCredentials(
                access_token=square_token
            ),
            environment='production'
        )
        
        # Fetch invoices for the location
        invoices_list = []
        cursor = None
        while True:
            response = client.invoices.list_invoices(location_id=location_id, cursor=cursor)
            if response.is_success():
                invoices = response.body.get('invoices', [])
                invoices_list.extend(invoices)
                cursor = response.body.get('cursor')
                if not cursor:
                    break  # Exit loop when no more pages
            elif response.is_error():
                print("Error:", response.errors)
                break

        # Filter for paid invoices and extract payment details
        paid_invoices = []
        for invoice in invoices_list:
            if invoice.get('status') == 'PAID':
                payment_requests = invoice.get('payment_requests', [])
                paid_date = None

                # Check each payment request for completed_at
                for request in payment_requests:
                    if 'completed_at' in request:  # Extract the first completed payment date
                        paid_date = request['completed_at']
                        break
                
                # Fallback: Use the invoice's updated_at field as an alternative date
                if not paid_date:
                    paid_date = invoice.get('updated_at')  # Use 'updated_at' as a fallback

                paid_invoice_amount = sum(
                    request.get('total_completed_amount_money', {}).get('amount', 0)
                    for request in payment_requests
                ) / 100  # Convert from cents to dollars

                paid_invoices.append({
                    'Description': 'paid rental invoice',
                    'Pre-Tax Amount': paid_invoice_amount / (1 + 0.0825) if paid_invoice_amount else 0,
                    'Tax Amount': paid_invoice_amount * 0.0825 if paid_invoice_amount else 0,
                    'Discount Amount': 0,
                    'Name': invoice.get('customer_id'),
                    'Total Amount': paid_invoice_amount,
                    'Date': paid_date,
                    'base_price_amount': 0,
                    'revenue_category': 'Event Booking',
                    'membership_size': None,
                    'membership_freq': None,
                    'is_founder': None,
                    'is_free_membership': None,
                    'date_': paid_date,
                    'Data Source': 'Square',
                    'Day Pass Count': None
                })

        # Create a DataFrame
        df = pd.DataFrame(paid_invoices)
        # Convert 'Date' to datetime for consistency
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
        # Extract just the date (without time)
        df['Date'] = df['Date'].dt.date

        return df

    def pull_and_transform_square_payment_data(start_date, end_date):
        # Get your Square Access Token from environment variables
        square_token = os.getenv('SQUARE_PRODUCTION_API_TOKEN')
        # Define the location ID
        location_id = "L37KDMNNG84EA"

        # Format the dates in ISO 8601 format
        end_time = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        begin_time = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Set the maximum limit to 1000
        limit = 1000
        df = pullSquareData.pull_square_payments_data_raw(square_token, location_id, end_time, begin_time, limit)
        df = pullSquareData.transform_payments_data(df)
        pullSquareData.save_data(df, 'square_transaction_data')
        # separate API call for paid invoices through Square
        invoices_df = pullSquareData.pull_square_invoices(square_token, location_id)
        pullSquareData.save_data(invoices_df, 'square_invoices_data')
        # combine
        df_combined = pd.concat([df, invoices_df], ignore_index=True)
        pullSquareData.save_data(df_combined, 'square_combined_transaction_invoices_data')
        return df_combined


if __name__ == "__main__":
    # Get today's date and calculate the start date for the last year
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)
    pull_square = pullSquareData()
    pull_square.pull_and_transform_square_payment_data(start_date, end_date)
