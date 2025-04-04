import streamlit as st
import pandas as pd
import os
from agent.analyze import analyze_trends_and_anomalies
from agent.explain import explain_anomalies, save_user_explanation

st.set_page_config(page_title="Trend Agent", layout="wide")
st.title("📊 Basin Revenue Trend Agent")

# === Load data ===
data_path = "data/outputs/combined_transaction_data.csv"
if not os.path.exists(data_path):
    st.error("❌ Data file not found at 'data/outputs/combined_transaction_data.csv'")
    st.stop()

st.success("✅ Payment data loaded.")
df = pd.read_csv(data_path, parse_dates=["Date"])

# === Sidebar: Upload user explanation ===
st.sidebar.header("📝 Explain Anomaly")
explanation_date = st.sidebar.date_input("Anomaly Date")
explanation_text = st.sidebar.text_area("What happened on this day?", placeholder="e.g. School holiday, special event, promotion...")

if st.sidebar.button("💾 Save Explanation"):
    if explanation_text:
        save_user_explanation(explanation_date.strftime("%Y-%m-%d"), explanation_text)
        st.sidebar.success("✅ Explanation saved!")
    else:
        st.sidebar.warning("Please write an explanation before saving.")

# === Main Panel ===
st.header("🔍 Trend Analysis")
summary = analyze_trends_and_anomalies(df)
st.text_area("📋 Summary of Findings", summary, height=200)

# === Agent Explanation ===
st.header("🤖 AI Agent: Explain Anomalies")
user_context = st.text_input("Optional: Add context (e.g. local events, promos, holidays)")

if st.button("🧠 Generate AI Explanation"):
    with st.spinner("Asking the AI agent for insights..."):
        ai_response = explain_anomalies(df, user_context)
        st.markdown("### 💬 AI Agent Response")
        st.write(ai_response)
