import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load env vars like your API key

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_openai(prompt: str, model="gpt-4", temperature=0.3) -> str:
    """Send a prompt to OpenAI's chat model and return the response text."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant analyzing business data for a climbing gym. Give insightful and actionable feedback."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI API error: {e}"