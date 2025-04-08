import sys
sys.path.append('./src')
from dash import Dash
# from oldDashboard import create_dashboard  # Import the function to set up the dashboard
# from dashboard import create_dashboard  # Import the function to set up the dashboard
from dashboard import create_dashboard
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run the Basin Climbing Dashboard')
parser.add_argument('--use-cache', action='store_true', help='Use cached data instead of fetching fresh data')
args = parser.parse_args()

# Initialize the Dash app
app = Dash(__name__)

# Create the dashboard layout and callbacks
create_dashboard(app, use_cached_data=args.use_cache)

# Expose the Flask server for Gunicorn
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)