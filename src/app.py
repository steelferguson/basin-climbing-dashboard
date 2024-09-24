from dash import Dash
from src.dashboard import app  # Import the layout and callbacks from dashboard

# Expose the Flask server to Gunicorn
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)