from dash import html, dcc, dash_table, Input, Output
import plotly.express as px
from pullSquareAndStripeData import pullSquareAndStripeData as squareAndStripe
from pullDataFromCapitan import pullDataFromCapitan as capitan
import pandas as pd

# Get the transformed data from Capitan
df = capitan.pull_and_transform_payment_data()
# df = pd.read_csv('data/outputs/payment_data.csv')

# Get the combined Stripe and Square data
df_combined = squareAndStripe.pull_and_transform_square_and_stripe_data()
# df_combined = pd.read_csv('data/outputs/square_transaction_data.csv')  # Path to your CSV file

# Define the layout and callbacks for the app here (don't define the app object itself)
def create_dashboard(app):
    # App layout
    app.layout = html.Div([
        html.H1(children='CAPITAN Revenue by Category Over Time'),

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

        # Line Graph for Capitan data
        dcc.Graph(id='line-chart'),

        # Add a new section for Square and Stripe Revenue
        html.H1(children='Square and Stripe Revenue By Category Over Time'),

        # Add the source toggle for Square and Stripe data
        dcc.Checklist(
            id='source-toggle',
            options=[
                {'label': 'Square', 'value': 'Square'},
                {'label': 'Stripe', 'value': 'Stripe'}
            ],
            value=['Square', 'Stripe'],  # Default is both selected
            inline=True
        ),

        # New Line Graph for Square and Stripe Revenue
        dcc.Graph(id='square-stripe-revenue-chart')
    ])

    # Callback to update the Capitan data and chart
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

    # Add a source toggle for Square and Stripe
    dcc.Checklist(
        id='source-toggle',
        options=[
            {'label': 'Square', 'value': 'Square'},
            {'label': 'Stripe', 'value': 'Stripe'}
        ],
        value=['Square', 'Stripe'],  # Default is both selected
        inline=True
    ),

    # New callback to update the Square and Stripe Revenue chart with the selected source and timeframe
    @app.callback(
        Output('square-stripe-revenue-chart', 'figure'),
        [Input('timeframe-toggle', 'value'),
        Input('source-toggle', 'value')]
    )
    def update_square_stripe_chart(selected_timeframe, selected_sources):
        # Filter by selected sources
        df_filtered = df_combined[df_combined['Data Source'].isin(selected_sources)]

        # Resample the combined data based on the selected time granularity
        # Convert 'Date' to datetime
        df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')

        # Then use the .dt accessor
        df_filtered['date'] = df_filtered['Date'].dt.to_period(selected_timeframe).dt.start_time
        df_filtered['date'] = df_filtered['Date'].dt.to_period(selected_timeframe).dt.start_time

        # Group by the new 'date' and 'revenue_category' and sum the 'Pre-Tax Amount'
        revenue_by_category = df_filtered.groupby(['date', 'revenue_category'])['Pre-Tax Amount'].sum().reset_index()

        # Create the figure
        fig = px.line(revenue_by_category, x='date', y='Pre-Tax Amount', color='revenue_category',
                    title='Square and Stripe Revenue By Category Over Time')

        return fig
    
# if __name__ == '__main__':
#     app.run_server(debug=True)