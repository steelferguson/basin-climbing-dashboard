from dash import Dash, html

# Initialize the Dash app
app = Dash(__name__)

# Set a simple Dash layout
app.layout = html.Div("Hello, Dash only!")

# Expose the Flask server for Gunicorn (Dash automatically creates one)
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)


# from dash import Dash, html

# # Initialize the Dash app
# app = Dash(__name__)

# # Set the layout using a Dash component
# app.layout = html.Div("Hello, Dash!")

# # Expose the Flask server for Heroku
# server = app.server

# if __name__ == "__main__":
#     app.run_server(debug=True)