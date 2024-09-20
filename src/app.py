import html
from dash import Dash

# Initialize the Dash app
app = Dash(__name__)

# You can set any server configurations here
# server = app.server

# Set the layout using a Dash component
app.layout = html.Div("Hello, World!")

server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)

# # Optional: Add external stylesheets, meta tags, etc.
# app.title = "Basin Climbing Dashboard"