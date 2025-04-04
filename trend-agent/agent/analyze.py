import pandas as pd
import numpy as np
import os
from datetime import datetime

def analyze_trends_and_anomalies(df):
    # Step 2: Preprocess
    df = df.dropna(subset=["Date", "Total Amount", "revenue_category"])
    df = df[df["Total Amount"] > 0]

    # Step 3: Aggregate Revenue by Day and Category
    daily_revenue = df.groupby(["Date", "revenue_category"])["Total Amount"].sum().reset_index()
    daily_total = df.groupby("Date")["Total Amount"].sum().reset_index(name="total_revenue")

    # Step 4: Detect anomalies with simple IQR filter
    Q1 = daily_total["total_revenue"].quantile(0.25)
    Q3 = daily_total["total_revenue"].quantile(0.75)
    IQR = Q3 - Q1
    threshold_low = Q1 - 1.5 * IQR
    threshold_high = Q3 + 1.5 * IQR
    daily_total["anomaly"] = daily_total["total_revenue"].apply(lambda x: x < threshold_low or x > threshold_high)

    # Step 5: Generate summary
    summary_lines = []
    summary_lines.append(f"Total date range: {daily_total['Date'].min().date()} to {daily_total['Date'].max().date()}")
    summary_lines.append(f"Average daily revenue: ${round(daily_total['total_revenue'].mean(), 2)}")

    max_day = daily_total.loc[daily_total["total_revenue"].idxmax()]
    min_day = daily_total.loc[daily_total["total_revenue"].idxmin()]
    summary_lines.append(f"Max revenue day: {max_day['Date'].date()} (${max_day['total_revenue']:.2f})")
    summary_lines.append(f"Min revenue day: {min_day['Date'].date()} (${min_day['total_revenue']:.2f})")
    summary_lines.append(f"Number of anomalous days: {daily_total['anomaly'].sum()}")

    detected_anomalies = daily_total[daily_total["anomaly"] == True]
    if not detected_anomalies.empty:
        summary_lines.append("\n🚨 Detected Anomaly Days:")
        for _, row in detected_anomalies.iterrows():
            day_of_week = row['Date'].strftime('%A')
            summary_lines.append(f"{row['Date'].date()} ({day_of_week}) - ${row['total_revenue']:.2f}")

    return "\n".join(summary_lines)

# If you want to test this module standalone
if __name__ == "__main__":
    data_path = "data/outputs/combined_transaction_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print("✅ Loading payment data...")
    df = pd.read_csv(data_path, parse_dates=['Date'])

    summary = analyze_trends_and_anomalies(df)
    print("\n📊 Summary of Findings:\n")
    print(summary)