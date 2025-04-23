import sys
sys.path.append('./src')
from dash import html, dcc, dash_table, Input, Output
import plotly.express as px
from pullSquareAndStripeData import pullSquareAndStripeData as squareAndStripe
from pullDataFromCapitan import pullDataFromCapitan as capitan
from pullSquareData import pullSquareData as square
from pullStripeData import pullStripeData
from projections.membership_projections import pullCapitanMembershipData
import pandas as pd
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go
import json

def load_or_fetch_data(use_cached_data=False, use_json=False):
    """Load data from cache if available and use_cached_data is True, otherwise fetch fresh data."""
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache file paths
    cache_files = {
        'capitan': f'{cache_dir}/capitan_data.csv',
        'membership_metrics': f'{cache_dir}/membership_metrics.csv',
        'square_stripe': f'{cache_dir}/square_stripe_data.csv',
        'projections': f'{cache_dir}/projections_data.csv',
        'memberships': f'{cache_dir}/memberships.json'  # New cache file for membership data
    }
    
    # Check if all cache files exist and use_cached_data is True
    if use_cached_data and all(os.path.exists(f) for f in cache_files.values()):
        print("Using cached data...")
        # Load cached data
        df = pd.read_csv(cache_files['capitan'])
        df_membership = pd.read_csv(cache_files['membership_metrics'])
        df_combined = pd.read_csv(cache_files['square_stripe'])
        df_projections = pd.read_csv(cache_files['projections'])
        
        # Load cached membership data
        with open(cache_files['memberships'], 'r') as f:
            membership_data = json.load(f)
        
        # Convert date columns back to datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        df_membership['date'] = pd.to_datetime(df_membership['date'])
        df_combined['Date'] = pd.to_datetime(df_combined['Date'])
        df_projections['date'] = pd.to_datetime(df_projections['date'])
        
        return df, df_membership, df_combined, df_projections, membership_data
    else:
        print("Fetching fresh data...")
        if use_json:
            print("Using saved JSON files instead of API calls...")
            # Load Capitan data from JSON
            with open('data/raw_data/capitan_payments.json', 'r') as f:
                capitan_data = json.load(f)
            df = pd.DataFrame(capitan_data['results'])
            
            # Transform Capitan data
            capitan_instance = capitan()
            df = capitan_instance.transform_payments_data(df)
            df = df[df['created_at'] >= '2023-08-01']
            
            # Calculate membership metrics from the transformed data
            df_membership = capitan_instance.calculate_membership_metrics(df)
            df_membership = df_membership[df_membership['date'] >= '2023-08-01']
            
            # Load Square and Stripe data from JSON
            with open('data/raw_data/square_orders.json', 'r') as f:
                square_data = json.load(f)
            with open('data/raw_data/stripe_payments.json', 'r') as f:
                stripe_data = json.load(f)
            with open('data/raw_data/square_invoices.json', 'r') as f:
                square_invoices_data = json.load(f)
            
            # Transform the data
            square_instance = square()
            stripe_instance = pullStripeData()
            
            # Handle Square orders data
            df_square = square_instance.create_orders_dataframe(square_data['orders'])
            df_square = square_instance.transform_payments_data(df_square)
            
            # Handle Stripe data
            df_stripe = stripe_instance.create_stripe_dataframe(stripe_data)
            print(f"Stripe data shape after creation: {df_stripe.shape}")
            print(f"Stripe data columns: {df_stripe.columns}")
            print(f"Stripe data sample:\n{df_stripe.head()}")
            
            df_stripe = stripe_instance.transform_payments_data(df_stripe)
            print(f"Stripe data shape after transformation: {df_stripe.shape}")
            print(f"Stripe data columns after transformation: {df_stripe.columns}")
            print(f"Stripe data sample after transformation:\n{df_stripe.head()}")
            
            # Handle Square invoices data
            df_square_invoices = square_instance.create_invoices_dataframe(square_invoices_data['invoices'])
            df_square_invoices = square_instance.transform_payments_data(df_square_invoices)
            
            # Combine all data
            df_combined = pd.concat([df_square, df_stripe, df_square_invoices], ignore_index=True)
            df_combined['Date'] = pd.to_datetime(df_combined['Date'], errors='coerce', utc=True)
            df_combined = df_combined[df_combined['Date'] >= '2023-08-01']
            
            # Load membership data from JSON
            with open('data/raw_data/capitan_customer_memberships.json', 'r') as f:
                membership_data = json.load(f)
            
            # Create projections from the loaded data
            projections, membership_summary = pullCapitanMembershipData.create_comprehensive_projection()
            
            # Convert projections to DataFrame
            projection_rows = []
            today = datetime.now()
            five_months_later = today + timedelta(days=150)  # 5 months for complete data
            
            for date, charges in projections.items():
                date_obj = pd.to_datetime(date)
                if today <= date_obj <= five_months_later:
                    for charge in charges:
                        projection_rows.append({
                            'date': date_obj,
                            'amount': charge['amount'],
                            'customer': charge['customer'],
                            'frequency': charge['categories']['frequency'],
                            'type': charge['categories']['type'],
                            'has_fitness': charge['categories']['has_fitness']
                        })
            
            df_projections = pd.DataFrame(projection_rows)
            
            # Export current membership data for comparison
            export_membership_data(membership_data)
            
            # Save to cache
            if use_cached_data:
                print("Saving data to cache...")
                df.to_csv(cache_files['capitan'], index=False)
                df_membership.to_csv(cache_files['membership_metrics'], index=False)
                df_combined.to_csv(cache_files['square_stripe'], index=False)
                df_projections.to_csv(cache_files['projections'], index=False)
                
                # Save membership data to cache
                with open(cache_files['memberships'], 'w') as f:
                    json.dump(membership_data, f)
        else:
            # Original code for fetching fresh data from APIs
            capitan_instance = capitan()
            df = capitan_instance.pull_and_transform_payment_data()
            df_membership = capitan_instance.calculate_membership_metrics(df)
            df_membership = df_membership[df_membership['date'] >= '2023-08-01']
            df = df[df['created_at'] >= '2023-08-01']
            
            df_combined = squareAndStripe.pull_and_transform_square_and_stripe_data(use_cached_data=use_cached_data)
            df_combined['Date'] = pd.to_datetime(df_combined['Date'], errors='coerce', utc=True)
            df_combined = df_combined[df_combined['Date'] >= '2023-08-01']
            
            projections, membership_summary = pullCapitanMembershipData.create_comprehensive_projection()
            
            # Convert projections to DataFrame
            projection_rows = []
            today = datetime.now()
            five_months_later = today + timedelta(days=150)  # 5 months for complete data
            
            for date, charges in projections.items():
                date_obj = pd.to_datetime(date)
                if today <= date_obj <= five_months_later:
                    for charge in charges:
                        projection_rows.append({
                            'date': date_obj,
                            'amount': charge['amount'],
                            'customer': charge['customer'],
                            'frequency': charge['categories']['frequency'],
                            'type': charge['categories']['type'],
                            'has_fitness': charge['categories']['has_fitness']
                        })
            
            df_projections = pd.DataFrame(projection_rows)
            
            # Get fresh membership data
            membership_data = pullCapitanMembershipData.get_memberships()
            
            # Export current membership data for comparison
            export_membership_data(membership_data)
            
            # Save to cache
            if use_cached_data:
                print("Saving data to cache...")
                df.to_csv(cache_files['capitan'], index=False)
                df_membership.to_csv(cache_files['membership_metrics'], index=False)
                df_combined.to_csv(cache_files['square_stripe'], index=False)
                df_projections.to_csv(cache_files['projections'], index=False)
                
                # Save membership data to cache
                with open(cache_files['memberships'], 'w') as f:
                    json.dump(membership_data, f)
        
        return df, df_membership, df_combined, df_projections, membership_data

# Define the layout and callbacks for the app here
def create_dashboard(app, use_cached_data=False, use_json=False):
    # Load or fetch data
    df, df_membership, df_combined, df_projections, membership_data = load_or_fetch_data(use_cached_data, use_json)
    
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
            inline=True,
            style={
                'backgroundColor': '#213B3F',
                'padding': '10px',
                'borderRadius': '5px',
                'color': '#FFFFFF',
                'marginBottom': '20px'
            }
        ),

        # Total Revenue chart section
        html.H1(children='Total Revenue Over Time',
                style={'color': '#213B3F', 'marginTop': '30px'}),
        dcc.Graph(id='total-revenue-chart'),

        # Square and Stripe Revenue section
        html.Div([
            html.H1(children='Square and Stripe Revenue Analysis',
                    style={'color': '#213B3F', 'marginTop': '30px'}),
            dcc.Checklist(
                id='source-toggle',
                options=[
                    {'label': 'Square', 'value': 'Square'},
                    {'label': 'Stripe', 'value': 'Stripe'}
                ],
                value=['Square', 'Stripe'],
                inline=True,
                style={'marginBottom': '20px'}
            ),
            dcc.Graph(id='square-stripe-revenue-chart'),
            dcc.Graph(id='square-stripe-revenue-stacked-chart'),
            dcc.Graph(id='revenue-percentage-chart'),
        ], style={'marginBottom': '40px'}),

        # Day Pass Count chart section
        html.H1(children='Day Pass Count',
                style={'color': '#213B3F', 'marginTop': '30px'}),
        dcc.Graph(id='day-pass-count-chart'),

        # Membership Revenue Projection chart section
        html.H1(children='Membership Revenue Projections (Current Month + 3 Months)',
                style={'color': '#213B3F', 'marginTop': '30px'}),
        html.Div([
            html.H3("Membership Frequency:"),
            dcc.Checklist(
                id='projection-frequency-toggle',
                options=[
                    {'label': 'Annual', 'value': 'yearly'},
                    {'label': 'Monthly', 'value': 'monthly'},
                    {'label': 'Bi-Weekly', 'value': 'bi_weekly'},
                    {'label': 'Prepaid', 'value': 'prepaid'}
                ],
                value=['yearly', 'monthly', 'bi_weekly', 'prepaid'],
                inline=True,
                style={'margin-bottom': '20px'}
            ),
            html.H3("Show Total Line:"),
            dcc.Checklist(
                id='show-total-toggle',
                options=[
                    {'label': 'Show Total', 'value': 'show_total'}
                ],
                value=['show_total'],
                inline=True
            )
        ]),
        dcc.Graph(id='membership-revenue-projection-chart'),

        # Membership Timeline chart section
        html.H1(children='Membership Timeline',
                style={'color': '#213B3F', 'marginTop': '30px'}),
        html.Div([
            html.H3("Membership Status:"),
            dcc.Checklist(
                id='status-toggle',
                options=[
                    {'label': 'Active', 'value': 'ACT'},
                    {'label': 'Ended', 'value': 'END'},
                    {'label': 'Frozen', 'value': 'FRZ'}
                ],
                value=['ACT'],  # Default to active only
                inline=True,
                style={'margin-bottom': '20px'}
            ),
            html.H3("Membership Frequency:"),
            dcc.Checklist(
                id='frequency-toggle',
                options=[
                    {'label': 'Bi-Weekly', 'value': 'bi_weekly'},
                    {'label': 'Monthly', 'value': 'monthly'},
                    {'label': 'Annual', 'value': 'annual'},
                    {'label': '3 Month Prepaid', 'value': 'prepaid_3mo'},
                    {'label': '6 Month Prepaid', 'value': 'prepaid_6mo'},
                    {'label': '12 Month Prepaid', 'value': 'prepaid_12mo'}
                ],
                value=['bi_weekly', 'monthly', 'annual', 'prepaid_3mo', 'prepaid_6mo', 'prepaid_12mo'],
                inline=True,
                style={'margin-bottom': '20px'}
            ),
            html.H3("Membership Size:"),
            dcc.Checklist(
                id='size-toggle',
                options=[
                    {'label': 'Solo', 'value': 'solo'},
                    {'label': 'Duo', 'value': 'duo'},
                    {'label': 'Family', 'value': 'family'}
                ],
                value=['solo', 'duo', 'family'],
                inline=True,
                style={'margin-bottom': '20px'}
            ),
            html.H3("Special Categories:"),
            dcc.Checklist(
                id='category-toggle',
                options=[
                    {'label': 'Founder', 'value': 'founder'},
                    {'label': 'College', 'value': 'college'},
                    {'label': 'Corporate', 'value': 'corporate'},
                    {'label': 'Mid-Day', 'value': 'mid_day'},
                    {'label': 'Fitness Only', 'value': 'fitness_only'},
                    {'label': 'Team Dues', 'value': 'team_dues'},
                    {'label': 'Include BCF Staff', 'value': 'include_bcf'}
                ],
                value=['founder', 'college', 'corporate', 'mid_day', 'fitness_only', 'team_dues'],
                inline=True,
                style={'margin-bottom': '20px'}
            ),
        ]),
        dcc.Graph(id='membership-timeline-chart'),

        # Youth Teams section
        html.H1(children='Youth Teams Membership',
                style={'color': '#213B3F', 'marginTop': '30px'}),
        dcc.Graph(id='youth-teams-chart'),
    ], style={
        'margin': '0 auto',
        'maxWidth': '1200px',
        'padding': '20px',
        'backgroundColor': '#FFFFFF',
        'color': '#26241C',
        'fontFamily': 'Arial, sans-serif'
    })

    # Update the color scheme for all charts
    chart_colors = {
        'primary': '#AF5436',    # rust
        'secondary': '#E9C867',  # gold
        'tertiary': '#BCCDA3',   # sage
        'quaternary': '#213B3F', # dark teal
        'background': '#F5F5F5', # light grey background
        'text': '#26241C'       # dark grey
    }

    # Define a sequence of colors for categorical data
    categorical_colors = [
        chart_colors['primary'],    # rust
        chart_colors['secondary'],  # gold
        chart_colors['tertiary'],   # sage
        chart_colors['quaternary'], # dark teal
        '#8B4229',  # darker rust
        '#BAA052',  # darker gold
        '#96A682',  # darker sage
        '#1A2E31'   # darker teal
    ]

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
        fig.update_traces(line_color=chart_colors['primary'])
        fig.update_layout(
            plot_bgcolor=chart_colors['background'],
            paper_bgcolor=chart_colors['background'],
            font_color=chart_colors['text']
        )
        return fig

    # Callback for Square and Stripe charts
    @app.callback(
        [Output('square-stripe-revenue-chart', 'figure'),
         Output('square-stripe-revenue-stacked-chart', 'figure'),
         Output('revenue-percentage-chart', 'figure')],
        [Input('timeframe-toggle', 'value'),
         Input('source-toggle', 'value')]
    )
    def update_square_stripe_charts(selected_timeframe, selected_sources):
        # Define revenue category colors and order
        revenue_category_colors = {
            'New Membership': chart_colors['secondary'],      # Gold
            'Membership Renewal': chart_colors['quaternary'], # Teal
            'Day Pass': chart_colors['primary'],             # Rust
            'Other': chart_colors['tertiary']                # Sage
        }

        # Define the order of categories
        category_order = ['New Membership', 'Membership Renewal', 'Day Pass', 'Other']

        # Filter and resample the Square and Stripe data
        df_filtered = df_combined[df_combined['Data Source'].isin(selected_sources)]
        df_filtered['date'] = df_filtered['Date'].dt.to_period(selected_timeframe).dt.start_time
        revenue_by_category = df_filtered.groupby(['date', 'revenue_category'])['Total Amount'].sum().reset_index()

        # Line chart
        line_fig = px.line(revenue_by_category, x='date', y='Total Amount', color='revenue_category',
                          title='Revenue By Category Over Time',
                          category_orders={'revenue_category': category_order})
        line_fig.update_layout(
            plot_bgcolor=chart_colors['background'],
            paper_bgcolor=chart_colors['background'],
            font_color=chart_colors['text']
        )
        for category in revenue_category_colors:
            line_fig.update_traces(line_color=revenue_category_colors[category], selector=dict(name=category))

        # Stacked column chart
        stacked_fig = px.bar(revenue_by_category, x='date', y='Total Amount', color='revenue_category',
                            title='Revenue (Stacked Column)', barmode='stack',
                            category_orders={'revenue_category': category_order})
        stacked_fig.update_layout(
            plot_bgcolor=chart_colors['background'],
            paper_bgcolor=chart_colors['background'],
            font_color=chart_colors['text']
        )
        for category in revenue_category_colors:
            stacked_fig.update_traces(marker_color=revenue_category_colors[category], selector=dict(name=category))

        # Percentage chart
        total_revenue_per_date = revenue_by_category.groupby('date')['Total Amount'].sum().reset_index()
        total_revenue_per_date.columns = ['date', 'total_revenue']
        revenue_with_total = pd.merge(revenue_by_category, total_revenue_per_date, on='date')
        revenue_with_total['percentage'] = (revenue_with_total['Total Amount'] / revenue_with_total['total_revenue']) * 100

        percentage_fig = px.bar(revenue_with_total, x='date', y='percentage', color='revenue_category',
                              title='Percentage of Revenue by Category', barmode='stack',
                              category_orders={'revenue_category': category_order})
        percentage_fig.update_layout(
            plot_bgcolor=chart_colors['background'],
            paper_bgcolor=chart_colors['background'],
            font_color=chart_colors['text']
        )
        for category in revenue_category_colors:
            percentage_fig.update_traces(marker_color=revenue_category_colors[category], selector=dict(name=category))

        return line_fig, stacked_fig, percentage_fig

    # Callback for Day Pass chart
    @app.callback(
        Output('day-pass-count-chart', 'figure'),
        [Input('timeframe-toggle', 'value')]
    )
    def update_day_pass_chart(selected_timeframe):
        df_filtered = df_combined[df_combined['revenue_category'] == 'Day Pass'].copy()
        df_filtered['date'] = df_filtered['Date'].dt.to_period(selected_timeframe).dt.start_time
        day_pass_sum = df_filtered.groupby('date')['Day Pass Count'].sum().reset_index(name='total_day_passes')

        fig = px.bar(day_pass_sum, x='date', y='total_day_passes', title='Total Day Passes Purchased')
        fig.update_traces(marker_color=chart_colors['quaternary'])  # Using teal color
        fig.update_layout(
            plot_bgcolor=chart_colors['background'],
            paper_bgcolor=chart_colors['background'],
            font_color=chart_colors['text']
        )
        return fig
        
    # Callback to update the Membership Revenue Projection chart
    @app.callback(
        Output('membership-revenue-projection-chart', 'figure'),
        [Input('timeframe-toggle', 'value'),
         Input('projection-frequency-toggle', 'value'),
         Input('show-total-toggle', 'value')]
    )
    def update_membership_revenue_projection_chart(selected_timeframe, selected_frequencies, show_total):
        if df_projections.empty:
            # If no data is available, create an empty figure with a message
            fig = px.bar(title='No membership projection data available')
            fig.add_annotation(
                text="No membership projection data available",
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            return fig
            
        # Calculate date range: remainder of current month + 3 complete months
        today = datetime.now().replace(tzinfo=None)  # Make timezone-naive
        # Get the last day of the current month
        if today.month == 12:
            end_of_current_month = datetime(today.year + 1, 1, 1, tzinfo=None) - timedelta(days=1)
        else:
            end_of_current_month = datetime(today.year, today.month + 1, 1, tzinfo=None) - timedelta(days=1)
        
        # Get the last day of the 3rd month from now
        if today.month + 3 > 12:
            end_of_third_month = datetime(today.year + 1, (today.month + 3) % 12, 1, tzinfo=None) - timedelta(days=1)
        else:
            end_of_third_month = datetime(today.year, today.month + 3, 1, tzinfo=None) - timedelta(days=1)
        
        # Get historical data from the current month
        start_of_current_month = datetime(today.year, today.month, 1, tzinfo=None)
        
        # Convert df_combined['Date'] to timezone-naive for comparison
        df_combined['Date'] = df_combined['Date'].dt.tz_localize(None)
        
        historical_data = df_combined[
            (df_combined['Date'] >= start_of_current_month) & 
            (df_combined['Date'] <= today) &
            (df_combined['revenue_category'].isin(['Membership Renewal', 'New Membership']))
        ].copy()
        
        # Filter projection data
        df_filtered = df_projections[
            (df_projections['date'] >= today) & 
            (df_projections['date'] <= end_of_third_month)
        ].copy()
        
        # Filter by selected frequencies
        filtered_rows = []
        for _, row in df_filtered.iterrows():
            # Check if this row should be included based on selected frequencies
            include_row = False
            
            # If this is a prepaid membership (3mo, 6mo, or 12mo)
            if row['frequency'] in ['prepaid_3mo', 'prepaid_6mo', 'prepaid_12mo']:
                if 'prepaid' in selected_frequencies:
                    include_row = True
            # For other frequencies, check if they're selected
            elif row['frequency'] in selected_frequencies:
                include_row = True
            
            if include_row:
                filtered_rows.append(row)
        
        df_filtered = pd.DataFrame(filtered_rows)
        
        # Resample the data based on selected timeframe
        df_filtered['date'] = df_filtered['date'].dt.to_period(selected_timeframe).dt.start_time
        historical_data['date'] = historical_data['Date'].dt.to_period(selected_timeframe).dt.start_time
        
        # Categorize each row by membership type
        def categorize_membership(row):
            if 'fitness only' in row['type'].lower():
                return 'Fitness Only'
            elif row['has_fitness']:
                return 'Climbing + Fitness'
            else:
                return 'Climbing Only'
        
        df_filtered['membership_type'] = df_filtered.apply(categorize_membership, axis=1)
        
        # Group by date and membership type for projections
        revenue_by_type = df_filtered.groupby(['date', 'membership_type'])['amount'].sum().reset_index()
        
        # Group historical data by date
        historical_revenue = historical_data.groupby('date')['Total Amount'].sum().reset_index()
        historical_revenue['membership_type'] = 'Historical'
        
        # Calculate total revenue for each date
        total_revenue = revenue_by_type.groupby('date')['amount'].sum().reset_index()
        
        # Add historical amounts to total_revenue where dates match
        for date in historical_revenue['date'].unique():
            if date in total_revenue['date'].values:
                idx = total_revenue[total_revenue['date'] == date].index[0]
                total_revenue.loc[idx, 'amount'] += historical_revenue[historical_revenue['date'] == date]['Total Amount'].iloc[0]
            else:
                new_row = pd.DataFrame({'date': [date], 'amount': [historical_revenue[historical_revenue['date'] == date]['Total Amount'].iloc[0]]})
                total_revenue = pd.concat([total_revenue, new_row], ignore_index=True)
        
        total_revenue = total_revenue.sort_values('date')
        total_revenue['membership_type'] = 'Total'
        
        # Create the figure
        fig = go.Figure()
        
        # Define colors for each membership type
        membership_colors = {
            'Climbing Only': chart_colors['quaternary'],  # Teal
            'Climbing + Fitness': chart_colors['primary'],  # Rust
            'Fitness Only': chart_colors['tertiary'],  # Sage
            'Historical': '#808080'  # Grey
        }

        # Add bars for each membership type
        for membership_type in revenue_by_type['membership_type'].unique():
            type_data = revenue_by_type[revenue_by_type['membership_type'] == membership_type]
            fig.add_trace(go.Bar(
                x=type_data['date'],
                y=type_data['amount'],
                name=membership_type,
                marker_color=membership_colors.get(membership_type, '#000000')
            ))
        
        # Add historical data as a separate bar
        fig.add_trace(go.Bar(
            x=historical_revenue['date'],
            y=historical_revenue['Total Amount'],
            name='Historical',
            marker_color=membership_colors['Historical']
        ))
        
        # Add total labels for all data points
        for i, row in total_revenue.iterrows():
            fig.add_annotation(
                x=row['date'],
                y=row['amount'],
                text=f"${row['amount']:,.2f}",
                showarrow=False,
                yshift=10
            )
        
        # Update layout
        fig.update_layout(
            title='Membership Revenue Projections (Current Month + 3 Months)',
            xaxis_title='Date',
            yaxis_title='Projected Revenue ($)',
            barmode='stack',
            showlegend=True,
            height=600,
            plot_bgcolor=chart_colors['background'],
            paper_bgcolor=chart_colors['background'],
            font_color=chart_colors['text']
        )
        
        # Always show month labels regardless of timeframe
        fig.update_xaxes(
            tickformat="%b %Y",
            dtick="M1"
        )
        
        return fig

    # Callback to update the Membership Timeline chart
    @app.callback(
        Output('membership-timeline-chart', 'figure'),
        [Input('frequency-toggle', 'value'),
         Input('size-toggle', 'value'),
         Input('category-toggle', 'value'),
         Input('status-toggle', 'value')]
    )
    def update_membership_timeline_chart(frequency_toggle, size_toggle, category_toggle, status_toggle):
        # Use the cached membership data
        print(f"Got membership data: {membership_data is not None}")
        if not membership_data:
            # If no data is available, create an empty figure with a message
            fig = px.bar(title='No membership data available')
            fig.add_annotation(
                text="No membership data available",
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            return fig
            
        # Create a new DataFrame for membership info
        membership_data_list = []
        unknown_frequencies = {}  # Track unknown frequencies for debugging
        
        # Process each membership
        for membership in membership_data.get('results', []):
            if membership.get('status') not in status_toggle:
                continue
                
            # Get frequency from interval and name
            interval = membership.get('interval', '').upper()
            name = str(membership.get('name', '')).lower()
            
            # Initialize all category flags
            is_founder = 'founder' in name
            is_college = 'college' in name
            is_corporate = 'corporate' in name or 'tfnb' in name or 'founders business' in name
            is_mid_day = 'mid-day' in name or 'mid day' in name
            is_fitness_only = 'fitness only' in name or 'fitness-only' in name
            has_fitness_addon = 'fitness' in name and not is_fitness_only
            is_team_dues = 'team dues' in name or 'team-dues' in name
            is_bcf = 'bcf' in name or 'staff' in name
            
            # Determine size
            if 'family' in name:
                size = 'family'
            elif 'duo' in name:
                size = 'duo'
            elif 'founders business' in name:
                size = 'corporate'
            else:
                size = 'solo'  # Default to solo if not specified
            
            # Check for specific membership types in the name
            if '3 month' in name or '3-month' in name:
                frequency = 'prepaid_3mo'
            elif '6 month' in name or '6-month' in name:
                frequency = 'prepaid_6mo'
            elif '12 month' in name or '12-month' in name:
                frequency = 'prepaid_12mo'
            elif is_mid_day:
                frequency = 'bi_weekly'
            elif is_bcf:  # Add BCF check before interval check
                frequency = 'bi_weekly'
            # Then check the interval
            elif interval == 'BWK':
                frequency = 'bi_weekly'
            elif interval == 'MON':
                frequency = 'monthly'
            elif interval == 'YRL' or interval == 'YEA':
                frequency = 'annual'
            elif interval == '3MO':
                frequency = 'prepaid_3mo'
            elif interval == '6MO':
                frequency = 'prepaid_6mo'
            elif interval == '12MO':
                frequency = 'prepaid_12mo'
            else:
                frequency = 'unknown'
                # Log details about unknown frequency
                key = f"{interval}_{membership.get('name', '')}"
                unknown_frequencies[key] = {
                    'interval': interval,
                    'name': membership.get('name'),
                    'start_date': membership.get('start_date'),
                    'end_date': membership.get('end_date'),
                    'billing_amount': membership.get('billing_amount')
                }
            
            # Get start and end dates
            start_date = pd.to_datetime(membership.get('start_date'), errors='coerce')
            end_date = pd.to_datetime(membership.get('end_date'), errors='coerce')
            
            # Skip memberships with invalid dates
            if pd.isna(start_date) or pd.isna(end_date):
                print(f"Skipping membership with invalid dates - Start: {membership.get('start_date')}, End: {membership.get('end_date')}")
                continue
            
            # Apply filters
            if frequency not in frequency_toggle:
                continue
            if size not in size_toggle:
                continue
            if is_bcf and 'include_bcf' not in category_toggle:
                continue
            if is_founder and 'founder' not in category_toggle:
                continue
            if is_college and 'college' not in category_toggle:
                continue
            if is_corporate and 'corporate' not in category_toggle:
                continue
            if is_mid_day and 'mid_day' not in category_toggle:
                continue
            if is_fitness_only and 'fitness_only' not in category_toggle:
                continue
            if is_team_dues and 'team_dues' not in category_toggle:
                continue
            
            # Debug print for BCF staff memberships
            if is_bcf:
                print(f"BCF Staff Membership: {membership.get('name')} - Included: {'include_bcf' in category_toggle}")
            
            membership_data_list.append({
                'customer_id': membership.get('customer_id'),
                'name': membership.get('name', ''),
                'frequency': frequency,
                'size': size,
                'is_founder': is_founder,
                'is_college': is_college,
                'is_corporate': is_corporate,
                'is_mid_day': is_mid_day,
                'is_fitness_only': is_fitness_only,
                'has_fitness_addon': has_fitness_addon,
                'is_team_dues': is_team_dues,
                'start_date': start_date,
                'end_date': end_date,
                'billing_amount': membership.get('billing_amount'),
                'interval': interval,
                'status': membership.get('status', '')
            })
        
        print(f"Processed {len(membership_data_list)} memberships")
        print("\nUnknown Frequencies Summary:")
        # Group unknown frequencies by interval and name pattern
        unknown_groups = {}
        for key, details in unknown_frequencies.items():
            group_key = f"{details['interval']}_{details['name'].lower()}"
            if group_key not in unknown_groups:
                unknown_groups[group_key] = {
                    'count': 0,
                    'interval': details['interval'],
                    'name': details['name'],
                    'total_amount': 0
                }
            unknown_groups[group_key]['count'] += 1
            unknown_groups[group_key]['total_amount'] += float(details['billing_amount'] or 0)
        
        # Print summary of unknown frequencies
        for group_key, group_data in unknown_groups.items():
            print(f"\n{group_data['name']} (Interval: {group_data['interval']})")
            print(f"Count: {group_data['count']}")
            print(f"Total Billing Amount: ${group_data['total_amount']:.2f}")
        
        if not membership_data_list:
            # If no valid memberships, create an empty figure with a message
            fig = px.bar(title='No valid membership data available')
            fig.add_annotation(
                text="No valid membership data available",
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create a new DataFrame from the membership data
        memberships_df = pd.DataFrame(membership_data_list)
        print(f"Created DataFrame with columns: {memberships_df.columns}")
        print(f"Date range: {memberships_df['start_date'].min()} to {memberships_df['end_date'].max()}")
        
        # Create a date range from the earliest start date to today
        min_date = memberships_df['start_date'].min()
        max_date = datetime.now()
        
        if pd.isna(min_date):
            print("Error: No valid start dates found in membership data")
            # If no valid start dates, create an empty figure with a message
            fig = px.bar(title='No valid start dates found')
            fig.add_annotation(
                text="No valid start dates found in membership data",
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        print(f"Created date range from {min_date} to {max_date}")
        
        # Calculate active memberships for each day by frequency
        daily_counts = []
        for date in date_range:
            # Count memberships that are active on this date
            active_memberships = memberships_df[
                (memberships_df['start_date'] <= date) & 
                (memberships_df['end_date'] >= date)
            ]
            
            # Count by frequency
            counts = active_memberships['frequency'].value_counts().to_dict()
            daily_counts.append({
                'date': date,
                'bi_weekly': counts.get('bi_weekly', 0),
                'monthly': counts.get('monthly', 0),
                'annual': counts.get('annual', 0),
                'prepaid_3mo': counts.get('prepaid_3mo', 0),
                'prepaid_6mo': counts.get('prepaid_6mo', 0),
                'prepaid_12mo': counts.get('prepaid_12mo', 0),
                'unknown': counts.get('unknown', 0)
            })
        
        daily_counts_df = pd.DataFrame(daily_counts)
        print(f"Created daily counts DataFrame with {len(daily_counts_df)} rows")
        print(f"Sample of daily counts: {daily_counts_df.head()}")
        
        # Create the line chart with updated colors
        fig = go.Figure()
        
        # Define colors for each frequency type
        frequency_colors = {
            'bi_weekly': chart_colors['primary'],
            'monthly': chart_colors['secondary'],
            'annual': chart_colors['tertiary'],
            'prepaid_3mo': '#8B4229',  # darker rust
            'prepaid_6mo': '#BAA052',  # darker gold
            'prepaid_12mo': '#96A682',  # darker sage
            'unknown': '#1A2E31'        # darker teal
        }
        
        # Add a line for each frequency type
        for frequency in ['bi_weekly', 'monthly', 'annual', 'prepaid_3mo', 'prepaid_6mo', 'prepaid_12mo', 'unknown']:
            if frequency in frequency_toggle:
                fig.add_trace(go.Scatter(
                    x=daily_counts_df['date'],
                    y=daily_counts_df[frequency],
                    mode='lines',
                    name=frequency.replace('_', ' ').title(),
                    stackgroup='one',
                    line=dict(color=frequency_colors[frequency])
                ))
        
        # Add total line
        total = daily_counts_df[['bi_weekly', 'monthly', 'annual', 'prepaid_3mo', 'prepaid_6mo', 'prepaid_12mo', 'unknown']].sum(axis=1)
        fig.add_trace(go.Scatter(
            x=daily_counts_df['date'],
            y=total,
            mode='lines',
            name='Total',
            line=dict(color=chart_colors['text'], width=2, dash='dash'),
            hovertemplate='Total: %{y}<extra></extra>'
        ))
        
        # Update layout for better readability with new theme
        fig.update_layout(
            title='Active Memberships Over Time by Payment Frequency',
            showlegend=True,
            height=600,
            xaxis_title='Date',
            yaxis_title='Number of Active Memberships',
            hovermode='x unified',
            plot_bgcolor=chart_colors['background'],
            paper_bgcolor=chart_colors['background'],
            font_color=chart_colors['text']
        )
        
        return fig

    # Callback for Youth Teams chart
    @app.callback(
        Output('youth-teams-chart', 'figure'),
        [Input('timeframe-toggle', 'value')]
    )
    def update_youth_teams_chart(selected_timeframe):
        if not membership_data:
            fig = px.bar(title='No youth teams data available')
            fig.add_annotation(
                text="No youth teams data available",
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            return fig

        # Create a list to store youth team memberships
        youth_memberships = []
        
        # Process each membership
        for membership in membership_data.get('results', []):
            name = str(membership.get('name', '')).lower()
            status = membership.get('status')
            
            # Only include active memberships
            if status != 'ACT':
                continue
            
            # Determine team type
            team_type = None
            if 'recreation' in name or 'rec team' in name:
                team_type = 'Recreation'
            elif 'development' in name or 'dev team' in name:
                team_type = 'Development'
            elif 'competitive' in name or 'comp team' in name:
                team_type = 'Competitive'
            
            if team_type:
                start_date = pd.to_datetime(membership.get('start_date'), errors='coerce')
                end_date = pd.to_datetime(membership.get('end_date'), errors='coerce')
                
                if not pd.isna(start_date) and not pd.isna(end_date):
                    youth_memberships.append({
                        'team_type': team_type,
                        'start_date': start_date,
                        'end_date': end_date
                    })
        
        if not youth_memberships:
            fig = px.bar(title='No youth teams data available')
            fig.add_annotation(
                text="No youth teams data available",
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create a DataFrame from youth memberships
        df_youth = pd.DataFrame(youth_memberships)
        
        # Create a date range from the earliest start date to today
        min_date = df_youth['start_date'].min()
        max_date = datetime.now()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Calculate active memberships for each day by team type
        daily_counts = []
        for date in date_range:
            active_memberships = df_youth[
                (df_youth['start_date'] <= date) & 
                (df_youth['end_date'] >= date)
            ]
            
            counts = active_memberships['team_type'].value_counts().to_dict()
            daily_counts.append({
                'date': date,
                'Recreation': counts.get('Recreation', 0),
                'Development': counts.get('Development', 0),
                'Competitive': counts.get('Competitive', 0)
            })
        
        daily_counts_df = pd.DataFrame(daily_counts)
        
        # Create the stacked line chart
        fig = go.Figure()
        
        # Define colors for each team type
        team_colors = {
            'Recreation': chart_colors['tertiary'],    # sage
            'Development': chart_colors['secondary'],  # gold
            'Competitive': chart_colors['primary']     # rust
        }
        
        # Add a line for each team type
        for team_type in ['Recreation', 'Development', 'Competitive']:
            fig.add_trace(go.Scatter(
                x=daily_counts_df['date'],
                y=daily_counts_df[team_type],
                mode='lines',
                name=team_type,
                stackgroup='one',
                line=dict(color=team_colors[team_type])
            ))
        
        # Add total line
        total = daily_counts_df[['Recreation', 'Development', 'Competitive']].sum(axis=1)
        fig.add_trace(go.Scatter(
            x=daily_counts_df['date'],
            y=total,
            mode='lines',
            name='Total',
            line=dict(color=chart_colors['text'], width=2, dash='dash'),
            hovertemplate='Total: %{y}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title='Youth Teams Membership Over Time',
            showlegend=True,
            height=600,
            xaxis_title='Date',
            yaxis_title='Number of Team Members',
            hovermode='x unified',
            plot_bgcolor=chart_colors['background'],
            paper_bgcolor=chart_colors['background'],
            font_color=chart_colors['text']
        )
        
        return fig

def export_membership_data(membership_data):
    """Export current membership data to CSV files for comparison."""
    # Create a new DataFrame for membership info
    membership_data_list = []
    detailed_membership_list = []
    
    # Process each membership
    for membership in membership_data.get('results', []):
        # Get frequency from interval and name
        interval = membership.get('interval', '').upper()
        name = str(membership.get('name', '')).lower()
        
        # Initialize all category flags
        is_founder = 'founder' in name
        is_college = 'college' in name
        is_corporate = 'corporate' in name or 'tfnb' in name or 'founders business' in name
        is_mid_day = 'mid-day' in name or 'mid day' in name
        is_fitness_only = 'fitness only' in name or 'fitness-only' in name
        has_fitness_addon = 'fitness' in name and not is_fitness_only
        is_team_dues = 'team dues' in name or 'team-dues' in name
        is_bcf = 'bcf' in name or 'staff' in name
        
        # Determine size
        if 'family' in name:
            size = 'family'
        elif 'duo' in name:
            size = 'duo'
        elif 'founders business' in name:
            size = 'corporate'
        else:
            size = 'solo'  # Default to solo if not specified
        
        # Check for specific membership types in the name
        if '3 month' in name or '3-month' in name:
            frequency = 'prepaid_3mo'
        elif '6 month' in name or '6-month' in name:
            frequency = 'prepaid_6mo'
        elif '12 month' in name or '12-month' in name:
            frequency = 'prepaid_12mo'
        elif is_mid_day:
            frequency = 'bi_weekly'
        elif is_bcf:  # Add BCF check before interval check
            frequency = 'bi_weekly'
        # Then check the interval
        elif interval == 'BWK':
            frequency = 'bi_weekly'
        elif interval == 'MON':
            frequency = 'monthly'
        elif interval == 'YRL' or interval == 'YEA':
            frequency = 'annual'
        elif interval == '3MO':
            frequency = 'prepaid_3mo'
        elif interval == '6MO':
            frequency = 'prepaid_6mo'
        elif interval == '12MO':
            frequency = 'prepaid_12mo'
        else:
            frequency = 'unknown'
        
        # Get start and end dates
        start_date = pd.to_datetime(membership.get('start_date'), errors='coerce')
        end_date = pd.to_datetime(membership.get('end_date'), errors='coerce')
        
        # Skip memberships with invalid dates
        if pd.isna(start_date) or pd.isna(end_date):
            continue
        
        # Add to summary list
        membership_data_list.append({
            'customer_id': membership.get('customer_id'),
            'name': membership.get('name', ''),
            'frequency': frequency,
            'size': size,
            'is_founder': is_founder,
            'is_college': is_college,
            'is_corporate': is_corporate,
            'is_mid_day': is_mid_day,
            'is_fitness_only': is_fitness_only,
            'has_fitness_addon': has_fitness_addon,
            'is_team_dues': is_team_dues,
            'start_date': start_date,
            'end_date': end_date,
            'billing_amount': membership.get('billing_amount'),
            'interval': interval,
            'status': membership.get('status', '')
        })
        
        # Add to detailed list
        detailed_membership_list.append({
            'customer_id': membership.get('customer_id'),
            'first_name': membership.get('owner_first_name', ''),
            'last_name': membership.get('owner_last_name', ''),
            'email': membership.get('owner_email', ''),
            'name': membership.get('name', ''),
            'frequency': frequency,
            'size': size,
            'is_founder': is_founder,
            'is_college': is_college,
            'is_corporate': is_corporate,
            'is_mid_day': is_mid_day,
            'is_fitness_only': is_fitness_only,
            'has_fitness_addon': has_fitness_addon,
            'is_team_dues': is_team_dues,
            'start_date': start_date,
            'end_date': end_date,
            'billing_amount': membership.get('billing_amount'),
            'interval': interval,
            'status': membership.get('status', '')
        })
    
    # Create DataFrames and export to CSV
    df = pd.DataFrame(membership_data_list)
    df.to_csv('data/cache/current_memberships.csv', index=False)
    print(f"Exported {len(df)} memberships to data/cache/current_memberships.csv")
    
    # Export detailed membership data
    detailed_df = pd.DataFrame(detailed_membership_list)
    detailed_df.to_csv('data/cache/detailed_memberships.csv', index=False)
    print(f"Exported {len(detailed_df)} detailed memberships to data/cache/detailed_memberships.csv")
    
    return df