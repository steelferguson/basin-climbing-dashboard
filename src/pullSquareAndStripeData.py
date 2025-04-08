import os
import datetime
import pandas as pd
from pullStripeData import pullStripeData as stripe
from pullSquareData import pullSquareData as square

class pullSquareAndStripeData:
    @staticmethod
    def pull_and_transform_square_and_stripe_data(use_cached_data=False):
        # Define cache file path
        cache_dir = 'data/cache'
        cache_file = f'{cache_dir}/square_stripe_data.csv'
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check if we should use cached data
        if use_cached_data and os.path.exists(cache_file):
            print("Using cached Square and Stripe data...")
            df_combined = pd.read_csv(cache_file)
            # Convert date column back to datetime
            df_combined['Date'] = pd.to_datetime(df_combined['Date'])
            return df_combined
            
        print("Fetching fresh Square and Stripe data...")
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365)
        df_stripe = stripe.pull_and_transform_stripe_payment_data(start_date, end_date)
        df_square = square.pull_and_transform_square_payment_data(start_date, end_date)
        df_combined = pd.concat([df_square, df_stripe], ignore_index=True)
        
        # Save to both cache and outputs
        df_combined.to_csv(cache_file, index=False)
        df_combined.to_csv('data/outputs/combined_transaction_data.csv', index=False)
        
        return df_combined

if __name__ == "__main__":
    pullSquareAndStripeData.pull_and_transform_square_and_stripe_data()