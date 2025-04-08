import sys
sys.path.append('./src')
from dash import html, dcc, dash_table, Input, Output
import plotly.express as px
from pullSquareAndStripeData import pullSquareAndStripeData as squareAndStripe
from pullDataFromCapitan import pullDataFromCapitan as capitan
from projections.membership_projections import pullCapitanMembershipData
import pandas as pd
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go
import json

def load_or_fetch_data(use_cached_data=False):
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
        # Get fresh data
        df = capitan.pull_and_transform_payment_data()
        df_membership = capitan.calculate_membership_metrics(df)
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
            if today <= date_obj <= five_months_later:  # Changed to five_months_later
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
def create_dashboard(app, use_cached_data=False):
    # Load or fetch data
    df, df_membership, df_combined, df_projections, membership_data = load_or_fetch_data(use_cached_data)
    
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

        # Total Revenue chart section
        html.H1(children='Total Revenue Over Time'),
        dcc.Graph(id='total-revenue-chart'),

        # Revenue Percentage chart section
        html.H1(children='Percentage of Revenue by Category'),
        dcc.Graph(id='revenue-percentage-chart'),

        # Day Pass Count chart section
        html.H1(children='Day Pass Count'),
        dcc.Graph(id='day-pass-count-chart'),  
        
        # Membership Count chart section
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
        
        # Membership Revenue Projection chart section
        html.H1(children='Membership Revenue Projections (Current Month + 4 Months)'),
        # Add checkboxes for membership frequency
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

        # Membership Timeline section
        html.H3('Membership Timeline'),
        html.Div([
            dcc.Checklist(
                id='exclude-bcf-toggle',
                options=[
                    {'label': 'Include BCF Staff Memberships', 'value': 'include_bcf'}
                ],
                value=[]  # Default to unchecked (exclude BCF)
            ),
        ], style={'margin-bottom': '20px'}),
        dcc.Graph(id='membership-timeline-chart'),
    ], style={'margin-top': '20px', 'padding': '20px', 'background-color': '#f8f9fa', 'border-radius': '5px'})

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

        # Get all columns that match the selected frequencies
        columns_to_sum = []
        for freq in selected_frequencies:
            # Convert frequency to match column naming convention
            if freq == 'bi_weekly':
                freq_prefix = 'weekly'
            elif freq == 'annual':
                freq_prefix = 'yearly'
            else:
                freq_prefix = freq
            
            # Add columns for each size
            for size in ['family', 'duo', 'solo']:
                col_name = f'{freq_prefix}_{size}'
                if col_name in df_filtered.columns:
                    columns_to_sum.append(col_name)

        if not columns_to_sum:
            # If no matching columns found, create an empty figure
            fig = px.bar(title='No membership data available')
            fig.add_annotation(
                text="No membership data available",
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            return fig

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
            
        # Calculate date range: remainder of current month + 4 complete months
        today = datetime.now().replace(tzinfo=None)  # Make timezone-naive
        # Get the last day of the current month
        if today.month == 12:
            end_of_current_month = datetime(today.year + 1, 1, 1, tzinfo=None) - timedelta(days=1)
        else:
            end_of_current_month = datetime(today.year, today.month + 1, 1, tzinfo=None) - timedelta(days=1)
        
        # Get the last day of the 4th month from now
        if today.month + 4 > 12:
            end_of_fourth_month = datetime(today.year + 1, (today.month + 4) % 12, 1, tzinfo=None) - timedelta(days=1)
        else:
            end_of_fourth_month = datetime(today.year, today.month + 4, 1, tzinfo=None) - timedelta(days=1)
        
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
            (df_projections['date'] <= end_of_fourth_month)
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
        
        # Resample the data to monthly intervals
        df_filtered['date'] = df_filtered['date'].dt.to_period('M').dt.start_time
        historical_data['date'] = historical_data['Date'].dt.to_period('M').dt.start_time
        
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
        total_revenue['membership_type'] = 'Total'
        
        # Create the figure
        fig = go.Figure()
        
        # Add bars for each membership type
        for membership_type in revenue_by_type['membership_type'].unique():
            type_data = revenue_by_type[revenue_by_type['membership_type'] == membership_type]
            fig.add_trace(go.Bar(
                x=type_data['date'],
                y=type_data['amount'],
                name=membership_type,
                marker_color={
                    'Climbing Only': '#1f77b4',  # Blue
                    'Climbing + Fitness': '#ff7f0e',  # Orange
                    'Fitness Only': '#2ca02c',  # Green
                    'Historical': '#888888'  # Gray
                }.get(membership_type, '#000000')
            ))
        
        # Add historical data as a separate bar
        fig.add_trace(go.Bar(
            x=historical_revenue['date'],
            y=historical_revenue['Total Amount'],
            name='Historical',
            marker_color='#888888'
        ))
        
        # Add total line if selected
        if 'show_total' in show_total:
            fig.add_trace(go.Scatter(
                x=total_revenue['date'],
                y=total_revenue['amount'],
                mode='lines+markers',
                name='Total',
                line=dict(color='black', width=2),
                marker=dict(size=8)
            ))
        
        # Add labels for all data points
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
            title='Membership Revenue Projections (Current Month + 4 Months)',
            xaxis_title='Date',
            yaxis_title='Projected Revenue ($)',
            barmode='stack',
            showlegend=True,
            height=600
        )
        
        # Update x-axis to show monthly intervals
        fig.update_xaxes(
            tickformat="%b %Y",
            dtick="M1"
        )
        
        return fig

    # Callback to update the Membership Timeline chart
    @app.callback(
        Output('membership-timeline-chart', 'figure'),
        [Input('exclude-bcf-toggle', 'value')]
    )
    def update_membership_timeline_chart(exclude_bcf):
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
            if membership.get('status') != 'ACT':
                continue
                
            # Include BCF staff memberships only if include_bcf is selected
            is_bcf = 'bcf' in str(membership.get('name', '')).lower() or 'staff' in str(membership.get('name', '')).lower()
            if is_bcf and 'include_bcf' not in exclude_bcf:
                continue
            
            # Get frequency from interval and name
            interval = membership.get('interval', '').upper()
            name = str(membership.get('name', '')).lower()
            
            # Check for specific membership types in the name
            if '3 month' in name or '3-month' in name:
                frequency = 'prepaid_3mo'
            elif '6 month' in name or '6-month' in name:
                frequency = 'prepaid_6mo'
            elif '12 month' in name or '12-month' in name:
                frequency = 'prepaid_12mo'
            elif 'mid-day' in name or 'mid day' in name:
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
            
            membership_data_list.append({
                'customer_id': membership.get('customer_id'),
                'frequency': frequency,
                'start_date': start_date,
                'end_date': end_date,
                'name': membership.get('name', '')
            })
        
        print(f"Processed {len(membership_data_list)} active memberships")
        print("\nUnknown Frequencies:")
        for key, details in unknown_frequencies.items():
            print(f"\nMembership: {details['name']}")
            print(f"Interval: {details['interval']}")
            print(f"Start Date: {details['start_date']}")
            print(f"End Date: {details['end_date']}")
            print(f"Billing Amount: {details['billing_amount']}")
        
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
        
        # Create the line chart
        fig = go.Figure()
        
        # Add a line for each frequency type
        for frequency in ['bi_weekly', 'monthly', 'annual', 'prepaid_3mo', 'prepaid_6mo', 'prepaid_12mo', 'unknown']:
            fig.add_trace(go.Scatter(
                x=daily_counts_df['date'],
                y=daily_counts_df[frequency],
                mode='lines',
                name=frequency.replace('_', ' ').title(),
                stackgroup='one'
            ))
        
        # Update layout for better readability
        fig.update_layout(
            title='Active Memberships Over Time by Payment Frequency',
            showlegend=True,
            height=600,
            xaxis_title='Date',
            yaxis_title='Number of Active Memberships',
            hovermode='x unified'
        )
        
        return fig