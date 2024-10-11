import sys
sys.path.append('./src')
from dash import html, dcc, dash_table, Input, Output
import plotly.express as px
from pullSquareAndStripeData import pullSquareAndStripeData as squareAndStripe
from pullDataFromCapitan import pullDataFromCapitan as capitan
import pandas as pd

# Get the transformed data from Capitan
df = capitan.pull_and_transform_payment_data()
df_membership = capitan.calculate_membership_metrics(df)
df_membership = df_membership[df_membership['date'] >= '2023-08-01']

# Set default start date to August 1
df = df[df['created_at'] >= '2023-08-01']

# Get the combined Stripe and Square data
df_combined = squareAndStripe.pull_and_transform_square_and_stripe_data()
df_combined['Date'] = pd.to_datetime(df_combined['Date'], errors='coerce', utc=True)
df_combined = df_combined[df_combined['Date'] >= '2023-08-01']

# Define the layout and callbacks for the app here
def create_dashboard(app):
    # App layout
    app.layout = html.Div([
        # Move time granularity toggle to the top
        dcc.RadioItems(
            id='timeframe-toggle',
            options=[
                {'label': 'Day', 'value': 'D'},
                {'label': 'Week', 'value': 'W'},
                {'label': 'Month', 'value': 'M'}
            ],
            value='W',  # Default to "Week"
            inline=True
        ),

        # Move Square and Stripe Revenue to the top
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

        # Line Graph for Square and Stripe Revenue
        dcc.Graph(id='square-stripe-revenue-chart'),

        # Stacked Column Chart for Square and Stripe Revenue
        dcc.Graph(id='square-stripe-revenue-stacked-chart'),

        html.H1(children='CAPITAN Revenue by Category Over Time'),

        # Data Table
        dash_table.DataTable(id='data-table', page_size=10),

        # Line Graph for Capitan data
        dcc.Graph(id='line-chart'),

        # New membership bar chart section
        html.H1(children='Membership Count by Frequency and Size'),
        
        # Checkboxes for membership size
        dcc.Checklist(
            id='membership-size-toggle',
            options=[
                {'label': 'Solo', 'value': 'solo'},
                {'label': 'Duo', 'value': 'duo'},
                {'label': 'Family', 'value': 'family'}
            ],
            value=['solo', 'duo', 'family'],  # Default is all selected
            inline=True
        ),

        # Bar chart for membership metrics
        dcc.Graph(id='membership-metrics-chart')
    ])

    # Callback to update the Capitan data and chart
    @app.callback(
        [Output('data-table', 'data'), Output('line-chart', 'figure')],
        [Input('timeframe-toggle', 'value')]
    )
    def update_capitan_graph(selected_timeframe):
        # Resample the data based on the selected time granularity
        df_filtered = df.copy()
        df_filtered['date'] = df_filtered['created_at'].dt.to_period(selected_timeframe).dt.start_time

        # Group by the new 'date' and 'revenue_category' and sum the 'amount_pre_tax'
        revenue_by_category = df_filtered.groupby(['date', 'revenue_category'])['amount_pre_tax'].sum().reset_index()

        # Create the figure and match colors with Square/Stripe chart
        fig = px.line(revenue_by_category, x='date', y='amount_pre_tax', color='revenue_category',
                      title=f'Revenue by Category ({selected_timeframe})')

        # Match line colors between Capitan and Square/Stripe chart
        fig.update_traces(marker=dict(line=dict(color='rgb(31, 119, 180)')))

        # Return data for table and figure for the graph
        return revenue_by_category.to_dict('records'), fig

    # Callback to update the Square and Stripe Revenue chart with the selected source and timeframe
    @app.callback(
        [Output('square-stripe-revenue-chart', 'figure'), Output('square-stripe-revenue-stacked-chart', 'figure')],
        [Input('timeframe-toggle', 'value'),
        Input('source-toggle', 'value')]
    )
    def update_square_stripe_charts(selected_timeframe, selected_sources):
        # Filter by selected sources
        df_filtered = df_combined[df_combined['Data Source'].isin(selected_sources)]

        # Convert 'Date' to datetime
        df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')

        # Resample the data based on the selected time granularity
        df_filtered['date'] = df_filtered['Date'].dt.to_period(selected_timeframe).dt.start_time

        # Group by 'date' and 'revenue_category' and sum the 'Pre-Tax Amount'
        revenue_by_category = df_filtered.groupby(['date', 'revenue_category'])['Pre-Tax Amount'].sum().reset_index()

        # Line chart
        line_fig = px.line(revenue_by_category, x='date', y='Pre-Tax Amount', color='revenue_category',
                           title='Square and Stripe Revenue By Category Over Time')

        # Stacked column chart
        stacked_fig = px.bar(revenue_by_category, x='date', y='Pre-Tax Amount', color='revenue_category',
                             title='Square and Stripe Revenue (Stacked Column)', barmode='stack')

        return line_fig, stacked_fig

    # Callback to update the membership metrics chart based on selected sizes
    @app.callback(
        Output('membership-metrics-chart', 'figure'),
        [Input('membership-size-toggle', 'value')]
    )
    def update_membership_metrics(selected_sizes):
        # Filter by selected membership sizes and sum up columns like yearly_solo, monthly_solo, etc.
        columns_to_sum = [f'yearly_{size}' for size in selected_sizes] + \
                         [f'monthly_{size}' for size in selected_sizes] + \
                         [f'weekly_{size}' for size in selected_sizes]

        df_filtered = df_membership.copy()
        df_filtered['total_memberships'] = df_filtered[columns_to_sum].sum(axis=1)

        # Create stacked bar chart for membership data
        fig = px.bar(df_filtered, x='date', y='total_memberships', title='Membership Count by Frequency and Size', barmode='stack')

        return fig