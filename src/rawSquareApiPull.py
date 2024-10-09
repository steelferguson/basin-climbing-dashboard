## Import everything; makes sure you have everything in your python environment
import pandas as pd
import requests
import json
import pprint

## JUST FOR LOCAL
# from dotenv import load_dotenv
import os 

# Load your Square access token from environment variables
# square_token = os.getenv('SQUARE_SANDBOX_API_TOKEN')
square_token = os.getenv('SQUARE_PRODUCTION_API_TOKEN')

print(square_token)

headers = {
    'Authorization': f'Bearer {square_token}',
    'Square-Version': '2024-09-19',  # Make sure to use the latest version
    'Content-Type': 'application/json'
}

print(headers)

# Define parameters (optional) to filter the response
params = {
    'begin_time': '2024-01-01T00:00:00Z',  # Start date for payments (ISO 8601 format)
    'end_time': '2024-12-31T23:59:59Z',    # End date for payments
}

# Make the GET request to the Square Payments endpoint
response = requests.get(
    'https://connect.squareup.com/v2/payments',
    headers=headers,
    params=params
)

if response.status_code == 200:
    # Process the revenue data
    data = response.json()
    pprint.pprint(data)
else:
    print('Error:', response.status_code, response.text)