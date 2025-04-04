from agent.analyze import analyze_trends_and_anomalies
from agent.explain import generate_agent_question, record_user_explanation
from utils.api_wrappers import ask_openai
import pandas as pd
import os

def run_trend_agent():
    # Step 1: Load data
    data_path = "data/outputs/combined_transaction_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print("✅ Loading payment data...")
    df = pd.read_csv(data_path, parse_dates=['Date'])

    # Step 2: Analyze trends and anomalies
    print("🔍 Analyzing trends and detecting anomalies...")
    summary = analyze_trends_and_anomalies(df)

    print("\n📊 Summary of Findings:\n")
    print(summary)

    # Step 3: Ask AI Agent for suggestions
    print("\n🤖 Generating suggestions using OpenAI...")
    agent_response = ask_openai(summary)

    print("\n💡 AI Agent Suggestions:\n")
    print(agent_response)

    # Step 4: Ask for human feedback on anomalies
    from agent.analyze import get_recent_anomalies
    anomalies = get_recent_anomalies()

    question = generate_agent_question(anomalies)
    if question:
        print("\n🧐 Agent Question:")
        print(question)
        explanation = input("💬 Your explanation: ")
        
        if explanation.strip():
            date_str = anomalies.sort_values("Date", ascending=False).iloc[0]["Date"].strftime('%Y-%m-%d')
            record_user_explanation(date_str, explanation)

if __name__ == "__main__":
    run_trend_agent()