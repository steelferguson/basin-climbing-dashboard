from dash import html, dcc, dash_table, Input, Output
# from app import app  # Import the initialized app
from src.app import app
import plotly.express as px
from pullDataFromCapitan import pull_and_transform_payment_data  # Import the function to pull data

# Get the transformed data from Capitan
df = pull_and_transform_payment_data()

# App layout
app.layout = html.Div([
    html.H1(children='Revenue by Category Over Time'),
    
    # Toggle for selecting time granularity (day, week, month)
    dcc.RadioItems(
        id='timeframe-toggle',
        options=[
            {'label': 'Day', 'value': 'D'},
            {'label': 'Week', 'value': 'W'},
            {'label': 'Month', 'value': 'M'}
        ],
        value='D',  # Default is daily
        inline=True
    ),
    
    # Filter for is_founder
    dcc.Checklist(
        id='is_founder-filter',
        options=[
            {'label': 'Founder', 'value': 'is_founder'}
        ],
        value=[],  # Default is no filter
        inline=True
    ),

    # Filter for is_free_membership
    dcc.Checklist(
        id='is_free-membership-filter',
        options=[
            {'label': 'Free Membership', 'value': 'is_free_membership'}
        ],
        value=[],  # Default is no filter
        inline=True
    ),

    # Data Table
    dash_table.DataTable(id='data-table', page_size=10),

    # Line Graph
    dcc.Graph(id='line-chart')
])

# Callback to update the data and chart based on the selected granularity and filters
@app.callback(
    [Output('data-table', 'data'), Output('line-chart', 'figure')],
    [Input('timeframe-toggle', 'value'),
     Input('is_founder-filter', 'value'),
     Input('is_free-membership-filter', 'value')]
)
def update_graph(selected_timeframe, is_founder_filter, is_free_membership_filter):
    # Resample the data based on the selected time granularity
    df_filtered = df.copy()

    # Apply is_founder filter if checked
    if 'is_founder' in is_founder_filter:
        df_filtered = df_filtered[df_filtered['is_founder'] == True]

    # Apply is_free_membership filter if checked
    if 'is_free_membership' in is_free_membership_filter:
        df_filtered = df_filtered[df_filtered['is_free_membership'] == True]
    
    # Resample the filtered data
    df_filtered['date'] = df_filtered['created_at'].dt.to_period(selected_timeframe).dt.start_time
    
    # Group by the new 'date' and 'revenue_category' and sum the 'amount_pre_tax'
    revenue_by_category = df_filtered.groupby(['date', 'revenue_category'])['amount_pre_tax'].sum().reset_index()
    
    # Create the figure
    fig = px.line(revenue_by_category, x='date', y='amount_pre_tax', color='revenue_category',
                  title=f'Revenue by Category ({selected_timeframe})')
    
    # Return data for table and figure for the graph
    return revenue_by_category.to_dict('records'), fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)