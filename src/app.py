from dash import Dash

# Initialize the Dash app
app = Dash(__name__)

# You can set any server configurations here
server = app.server

# Optional: Add external stylesheets, meta tags, etc.
app.title = "Basin Climbing Dashboard"