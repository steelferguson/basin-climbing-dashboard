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
    app.layout = html.Div([
        # Timeframe toggle
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

        # Square and Stripe Revenue section
        html.H1(children='Square and Stripe Revenue By Category Over Time'),
        dcc.Checklist(
            id='source-toggle',
            options=[
                {'label': 'Square', 'value': 'Square'},
                {'label': 'Stripe', 'value': 'Stripe'}
            ],
            value=['Square', 'Stripe'],  # Default is both selected
            inline=True
        ),
        dcc.Graph(id='square-stripe-revenue-chart'),
        dcc.Graph(id='square-stripe-revenue-stacked-chart'),

        # CAPITAN Revenue by Event Sub-Category section
        html.H1(children='CAPITAN Revenue by Sub-Category'),
        dcc.Graph(id='capitan-revenue-sub-category-chart'),

        # Total Revenue chart section
        html.H1(children='Total Revenue Over Time'),
        dcc.Graph(id='total-revenue-chart'),

        # Revenue Percentage chart section
        html.H1(children='Percentage of Revenue by Category'),
        dcc.Graph(id='revenue-percentage-chart'),

        # Day Pass Count chart section
        html.H1(children='Day Pass Count'),
        dcc.Graph(id='day-pass-count-chart'),  
        
        html.H1(children='Membership Count by Frequency and Size'),
        dcc.Checklist(
            id='membership-frequency-toggle',
            options=[
                {'label': 'Yearly', 'value': 'yearly'},
                {'label': 'Monthly', 'value': 'monthly'},
                {'label': 'Weekly', 'value': 'weekly'}
            ],
            value=['yearly', 'monthly', 'weekly'],  # Default is all selected
            inline=True
        ),
        dcc.Graph(id='membership-metrics-stacked-chart'),
    ])

    # Callback to update CAPITAN data and sub-category chart
    @app.callback(
        Output('capitan-revenue-sub-category-chart', 'figure'),
        [Input('timeframe-toggle', 'value')]
    )
    def update_capitan_sub_category_chart(selected_timeframe):
        # Set the start date to April 1
        start_date = pd.to_datetime('2024-04-01')

        # Filter for events only and set start date
        df_filtered = df[(df['revenue_category'] == 'Event Booking') & (df['created_at'] >= start_date)].copy()

        if df_filtered.empty:
            # If no data is available, create an empty figure with a message
            fig = px.bar(title='No data available for Event Revenue by Sub-Category')
            fig.add_annotation(
                text="No data available for the selected timeframe",
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            return fig

        # Resample and group the CAPITAN data by revenue_sub_category
        df_filtered['date'] = df_filtered['created_at'].dt.to_period(selected_timeframe).dt.start_time
        revenue_by_sub_category = df_filtered.groupby(['date', 'revenue_sub_category'])['amount'].sum().reset_index()

        # Create the stacked column chart
        fig = px.bar(revenue_by_sub_category, x='date', y='amount', color='revenue_sub_category',
                    title='Event Revenue by Sub-Category', barmode='stack')

        return fig

    # Callback for Total Revenue chart
    @app.callback(
        Output('total-revenue-chart', 'figure'),
        [Input('timeframe-toggle', 'value')]
    )
    def update_total_revenue_chart(selected_timeframe):
        df_filtered = df_combined.copy()
        df_filtered['date'] = df_filtered['Date'].dt.to_period(selected_timeframe).dt.start_time
        total_revenue = df_filtered.groupby('date')['Total Amount'].sum().reset_index()

        fig = px.line(total_revenue, x='date', y='Total Amount', title='Total Revenue Over Time')
        return fig

    # Callback for Percentage of Revenue by Category chart
    @app.callback(
        Output('revenue-percentage-chart', 'figure'),
        [Input('timeframe-toggle', 'value')]
    )
    def update_revenue_percentage_chart(selected_timeframe):
        df_filtered = df_combined.copy()
        df_filtered['date'] = df_filtered['Date'].dt.to_period(selected_timeframe).dt.start_time

        # Group by date and category and sum the revenue
        revenue_by_category = df_filtered.groupby(['date', 'revenue_category'])['Total Amount'].sum().reset_index()

        # Calculate the total revenue per date
        total_revenue_per_date = revenue_by_category.groupby('date')['Total Amount'].sum().reset_index()
        total_revenue_per_date.columns = ['date', 'total_revenue']

        # Merge to calculate percentages
        revenue_with_total = pd.merge(revenue_by_category, total_revenue_per_date, on='date')
        revenue_with_total['percentage'] = (revenue_with_total['Total Amount'] / revenue_with_total['total_revenue']) * 100

        fig = px.bar(revenue_with_total, x='date', y='percentage', color='revenue_category',
                     title='Percentage of Revenue by Category Over Time', barmode='stack')
        return fig

    # Callback to update membership metrics stacked bar chart
    @app.callback(
        Output('membership-metrics-stacked-chart', 'figure'),
        [Input('membership-frequency-toggle', 'value')]
    )
    def update_membership_stacked_chart(selected_frequencies):
        # Filter by selected membership frequencies
        df_filtered = df_membership.copy()

        columns_to_sum = [f'{freq}_{size}' for freq in selected_frequencies for size in ['family', 'duo', 'solo']]
        df_filtered['total_memberships'] = df_filtered[columns_to_sum].sum(axis=1)

        # Reshape the data for stacked bar chart
        membership_by_size = df_filtered.melt(id_vars='date', value_vars=columns_to_sum, var_name='membership_type', value_name='count')

        # Split membership_type into frequency and size for color coding
        membership_by_size[['frequency', 'size']] = membership_by_size['membership_type'].str.split('_', expand=True)

        # Create stacked bar chart for membership, with custom stacking order
        fig = px.bar(membership_by_size, x='date', y='count', color='size',
                    title='Membership Count by Size and Frequency', barmode='stack',
                    category_orders={'size': ['family', 'duo', 'solo']},
                    color_discrete_map={
                        'family': '#1f77b4',
                        'duo': '#ff7f0e',
                        'solo': '#2ca02c'
                    })

        # Add total count on top of the bars
        total_memberships_by_date = df_filtered.groupby('date')['total_memberships'].sum().reset_index()
        for i, row in total_memberships_by_date.iterrows():
            fig.add_annotation(x=row['date'], y=row['total_memberships'], text=str(row['total_memberships']),
                            showarrow=False, yshift=10, textangle=-60)

        return fig

    # Callback to update the Square and Stripe revenue charts
    @app.callback(
        [Output('square-stripe-revenue-chart', 'figure'), Output('square-stripe-revenue-stacked-chart', 'figure')],
        [Input('timeframe-toggle', 'value'), Input('source-toggle', 'value')]
    )
    def update_square_stripe_charts(selected_timeframe, selected_sources):
        # Filter and resample the Square and Stripe data
        df_filtered = df_combined[df_combined['Data Source'].isin(selected_sources)]
        df_filtered['date'] = df_filtered['Date'].dt.to_period(selected_timeframe).dt.start_time
        revenue_by_category = df_filtered.groupby(['date', 'revenue_category'])['Total Amount'].sum().reset_index()

        # Line chart
        line_fig = px.line(revenue_by_category, x='date', y='Total Amount', color='revenue_category',
                           title='Square and Stripe Revenue By Category Over Time')

        # Stacked column chart
        stacked_fig = px.bar(revenue_by_category, x='date', y='Total Amount', color='revenue_category',
                             title='Square and Stripe Revenue (Stacked Column)', barmode='stack')

        return line_fig, stacked_fig
    
    # Callback to update the Day Pass count chart
    @app.callback(
        Output('day-pass-count-chart', 'figure'),
        [Input('timeframe-toggle', 'value')]
    )
    def update_day_pass_chart(selected_timeframe):
        # Filter for rows where the revenue_category is 'Day Pass'
        df_filtered = df_combined[df_combined['revenue_category'] == 'Day Pass'].copy()

        # Resample and group by date to count the number of day passes
        df_filtered['date'] = df_filtered['Date'].dt.to_period(selected_timeframe).dt.start_time
        day_pass_sum = df_filtered.groupby('date')['Day Pass Count'].sum().reset_index(name='total_day_passes')

        # Create the bar chart
        fig = px.bar(day_pass_sum, x='date', y='total_day_passes', title='Total Day Passes Purchased')

        return fig