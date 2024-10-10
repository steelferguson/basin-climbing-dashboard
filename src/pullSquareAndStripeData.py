import os
import datetime
import pandas as pd
from pullStripeData import pullStripeData as stripe
from pullSquareData import pullSquareData as square

class pullSquareAndStripeData:
    @staticmethod
    def pull_and_transform_square_and_stripe_data():
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365)
        df_stripe = stripe.pull_and_transform_stripe_payment_data(start_date, end_date)
        df_square = square.pull_and_transform_square_payment_data(start_date, end_date)
        df_combined = pd.concat([df_square, df_stripe], ignore_index=True)
        df_combined.to_csv('data/outputs/combined_transaction_data.csv', index=False)
        return df_combined

if __name__ == "__main__":
    pullSquareAndStripeData.pull_and_transform_square_and_stripe_data()