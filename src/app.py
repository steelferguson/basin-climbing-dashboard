import sys
sys.path.append('./src')
from dash import Dash
from dashboard import create_dashboard
import os

# Get cache setting from environment variable instead of command line
use_cache = os.environ.get('USE_CACHE', 'false').lower() == 'true'

# Initialize the Dash app
app = Dash(__name__)

# Create the dashboard layout and callbacks
create_dashboard(app, use_cached_data=use_cache)

# Expose the Flask server for Gunicorn
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)