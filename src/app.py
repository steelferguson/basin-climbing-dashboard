import sys
sys.path.append('./src')
from dash import Dash
# from oldDashboard import create_dashboard  # Import the function to set up the dashboard
# from dashboard import create_dashboard  # Import the function to set up the dashboard
from dashboard import create_dashboard

# Initialize the Dash app
app = Dash(__name__)

# Create the dashboard layout and callbacks
create_dashboard(app)

# Expose the Flask server for Gunicorn
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)