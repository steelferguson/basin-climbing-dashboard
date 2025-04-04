import pandas as pd
from datetime import datetime
from pathlib import Path
import json
from agent.analyze import analyze_trends_and_anomalies

RESPONSES_FILE = Path("data/outputs/anomaly_explanations.json")

def explain_anomalies(df, additional_context: str = ""):
    """
    Use OpenAI to explain recent revenue anomalies and optionally suggest reasons or questions to ask.
    Args:
        df (pd.DataFrame): The cleaned dataframe of transactions.
        additional_context (str): Optional context from the user about known events.
    Returns:
        str: Agent's explanation and questions/suggestions.
    """
    summary, recent_anomalies = analyze_trends_and_anomalies(df, return_anomalies=True)

    if recent_anomalies.empty:
        return "✅ No recent revenue anomalies detected."

    # Create a summary string for OpenAI
    summary_lines = ["Here are recent revenue anomalies:"]
    for _, row in recent_anomalies.iterrows():
        summary_lines.append(f"- {row['Date'].date()} ({row['day_of_week']}): ${row['total_revenue']:.2f}")
    summary_text = "\n".join(summary_lines)

    system_prompt = f"""
You are a revenue trends analyst AI for a climbing gym. The user has shared recent anomaly days with high or low revenue.
{f"Additional context: {additional_context}" if additional_context else ""}
Given the anomalies listed below, generate a friendly explanation and ask 2-3 clarifying questions that could help explain the trends (e.g., holiday, event, promo).
"""

    full_prompt = f"{system_prompt}\n\n{summary_text}"

    from utils.api_wrappers import ask_openai
    return ask_openai(full_prompt)

def save_user_explanation(anomaly_date: str, user_response: str):
    """Store the user's explanation for a specific anomaly date."""
    if RESPONSES_FILE.exists():
        with open(RESPONSES_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[anomaly_date] = {
        "explanation": user_response,
        "timestamp": datetime.utcnow().isoformat()
    }

    with open(RESPONSES_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_all_user_explanations():
    """Load all saved user responses."""
    if RESPONSES_FILE.exists():
        with open(RESPONSES_FILE, "r") as f:
            return json.load(f)
    return {}

def get_context_for_anomalies(anomaly_dates):
    """Retrieve any saved user context related to specific anomaly dates."""
    all_context = get_all_user_explanations()
    return {
        date: all_context.get(date, {}).get("explanation")
        for date in anomaly_dates
    }