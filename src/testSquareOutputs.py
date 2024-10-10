from square.client import Client
from square.http.auth.o_auth_2 import BearerAuthCredentials
import os
import datetime
import pandas as pd
import pprint


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
        print("result: \n")
        pprint.pprint(result)
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

        # Break the loop if we already have 2 orders
        if len(orders_list) >= 2:
            break

    # Extract only the first 2 orders
    orders_list = orders_list[:2]
    
    # Extract relevant data for DataFrame
    data = []
    print("order list: \n\n")
    pprint.pprint(orders_list)
    for order in orders_list:
        created_at = order.get('created_at')  # Order creation date
        line_items = order.get('line_items', [])
        
        for item in line_items:
            name = item.get('name', 'No Name')
            description = item.get('variation_name', 'No Description')

            # Get the specific amount for each item
            item_total_money = item.get('total_money', {}).get('amount', 0) / 100  # Convert from cents
            item_pre_tax_money = item.get('variation_total_price_money', {}).get('amount', 0) / 100  # Pre-tax amount (if available)
            item_tax_money = item.get('total_tax_money', {}).get('amount', 0) / 100  # Tax amount for the item
            item_discount_money = item.get('total_discount_money', {}).get('amount', 0) / 100  # Discount for the item

            data.append({
                'Description': description,
                'Pre-Tax Amount': item_pre_tax_money,
                'Tax Amount': item_tax_money,
                'Discount Amount': item_discount_money,
                'Name': name,
                'Total Amount': item_total_money,
                'Date': created_at
            })
    # Create a DataFrame
    df = pd.DataFrame(data)
    return df

def test():
    # Get your Square Access Token from environment variables
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=1)
    square_token = os.getenv('SQUARE_PRODUCTION_API_TOKEN')
    # Define the location ID
    location_id = "L37KDMNNG84EA"

    # Format the dates in ISO 8601 format
    end_time = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    begin_time = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Set the maximum limit to 1000
    limit = 1000
    df = pull_square_payments_data_raw(square_token, location_id, end_time, begin_time, limit)

if __name__ == "__main__":
    # Get today's date and calculate the start date for the last year
    test()
    