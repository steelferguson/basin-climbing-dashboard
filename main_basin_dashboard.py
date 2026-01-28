"""
Basin Climbing & Fitness Dashboard - Streamlit Version

A comprehensive analytics dashboard for Basin Climbing & Fitness.
Organized into logical tabs for better navigation and analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_pipeline import upload_data
from data_pipeline import config
from data_pipeline import categorize_expenses
import os

# Page config
st.set_page_config(
    page_title="Basin Climbing Dashboard",
    page_icon="🧗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Basin brand colors
COLORS = {
    'primary': '#8B4229',      # Rust/terracotta
    'secondary': '#BAA052',    # Gold
    'tertiary': '#96A682',     # Sage green
    'quaternary': '#1A2E31',   # Dark teal
    'background': '#FFFFFF',
    'text': '#213B3F',
    'dark_grey': '#4A4A4A',
    'axis_text': '#2C2C2C',    # Dark grey for axis labels (easier to read)
    'gridline': '#E0E0E0'      # Light grey for gridlines (subtle)
}

# Set default Plotly template for consistent chart styling across all charts
import plotly.io as pio
import plotly.graph_objects as go

# Store axis styling config in one place - single source of truth
AXIS_CONFIG = {
    'tickfont': dict(color=COLORS['axis_text'], size=14),
    'gridcolor': COLORS['gridline'],
    'title_font': dict(color=COLORS['text'], size=16)
}

# Create custom Basin template - this will apply automatically to ALL charts
basin_template = go.layout.Template()
basin_template.layout.xaxis = AXIS_CONFIG
basin_template.layout.yaxis = AXIS_CONFIG
basin_template.layout.plot_bgcolor = COLORS['background']
basin_template.layout.paper_bgcolor = COLORS['background']
basin_template.layout.font = dict(color=COLORS['text'], size=14)
basin_template.layout.legend = dict(font=dict(color='#000000', size=15))

# Register and set as default
pio.templates['basin'] = basin_template
pio.templates.default = 'basin'

# Revenue category colors
REVENUE_CATEGORY_COLORS = {
    'Day Pass': COLORS['quaternary'],        # Dark teal
    'New Membership': COLORS['primary'],      # Rust
    'Membership Renewal': '#D4AF6A',          # Lighter gold (different from secondary)
    'Programming': COLORS['tertiary'],        # Sage green
    'Team Dues': '#C85A3E',                   # Lighter rust (different from primary)
    'Retail': COLORS['dark_grey'],            # Dark grey
    'Event Booking': '#B8C9A8',               # Lighter sage (different from tertiary)
}


def check_password():
    """
    Password protection for the dashboard.
    Returns True if the user has entered the correct password.
    """
    def password_entered():
        """Check if entered password is correct."""
        if st.session_state["password"] == os.getenv("DASHBOARD_PASSWORD", "basin2024"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # First run, show input for password
    if "password_correct" not in st.session_state:
        st.text_input(
            "🔒 Enter Dashboard Password",
            type="password",
            on_change=password_entered,
            key="password"
        )
        return False

    # Password not correct, show input + error
    elif not st.session_state["password_correct"]:
        st.text_input(
            "🔒 Enter Dashboard Password",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.error("😕 Password incorrect")
        return False

    # Password correct
    else:
        return True


def apply_axis_styling(fig):
    """
    Apply consistent axis styling to charts: darker labels, lighter gridlines.

    Call this after creating any chart to apply better readability styling.
    """
    fig.update_layout(
        xaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=16)
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=16)
        ),
        legend=dict(
            font=dict(color='#000000', size=15),
            title_font=dict(color='#000000', size=15)
        )
    )
    return fig


def convert_shopify_to_transactions(df_shopify: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Shopify orders to transactions dataframe format.

    Maps Shopify columns to match the transaction dataframe schema used by revenue tools.
    """
    if df_shopify.empty:
        return pd.DataFrame()

    # Map Shopify categories to dashboard revenue categories
    # Shopify uses "Birthday Party" but dashboard uses "Event Booking"
    category_map = {
        'Day Pass': 'Day Pass',
        'Birthday Party': 'Event Booking',
        'Other': 'Other'
    }
    mapped_category = df_shopify['category'].map(category_map).fillna('Other')

    # Create transactions dataframe from Shopify orders
    shopify_transactions = pd.DataFrame({
        'transaction_id': df_shopify['line_item_id'].astype(str),
        'Description': df_shopify['product_title'],
        'Pre-Tax Amount': df_shopify['price'] - (df_shopify['total_tax'] / df_shopify['quantity']),  # Approximate per-item tax
        'Tax Amount': df_shopify['total_tax'] / df_shopify['quantity'],  # Approximate per-item tax
        'Total Amount': df_shopify['price'],
        'Discount Amount': df_shopify['total_discounts'] / df_shopify['quantity'],  # Approximate per-item discount
        'Name': df_shopify['customer_first_name'].fillna('') + ' ' + df_shopify['customer_last_name'].fillna(''),
        'Date': df_shopify['transaction_date'],
        'payment_intent_status': 'succeeded',  # Shopify orders are completed
        'revenue_category': mapped_category,  # Mapped to dashboard categories
        'membership_size': None,
        'membership_freq': None,
        'is_founder': False,
        'is_free_membership': False,
        'sub_category': df_shopify['variant_title'],
        'sub_category_detail': df_shopify['product_title'],
        'date_': df_shopify['transaction_date'],
        'Data Source': 'Shopify',
        'Day Pass Count': df_shopify['quantity'].where(df_shopify['category'] == 'Day Pass', 0),
        'base_price_amount': df_shopify['price'],
        'status': df_shopify['financial_status'],
        'payment_id': df_shopify['order_id'].astype(str),
        'order_id': df_shopify['order_id'].astype(str),
        'quantity': df_shopify['quantity'],
        'fitness_amount': 0.0
    })

    return shopify_transactions


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load all data from S3 with caching."""
    uploader = upload_data.DataUploader()

    def load_df(bucket, key):
        try:
            csv_content = uploader.download_from_s3(bucket, key)
            return uploader.convert_csv_to_df(csv_content)
        except Exception:
            # Return empty DataFrame if file doesn't exist
            return pd.DataFrame()

    df_transactions = load_df(config.aws_bucket_name, config.s3_path_combined)
    df_memberships = load_df(config.aws_bucket_name, config.s3_path_capitan_memberships)
    df_members = load_df(config.aws_bucket_name, config.s3_path_capitan_members)
    df_projection = load_df(config.aws_bucket_name, config.s3_path_capitan_membership_revenue_projection)
    df_at_risk = load_df(config.aws_bucket_name, config.s3_path_at_risk_members)
    df_new_members = load_df(config.aws_bucket_name, config.s3_path_new_members)
    df_facebook_ads = load_df(config.aws_bucket_name, config.s3_path_facebook_ads)
    df_events = load_df(config.aws_bucket_name, config.s3_path_capitan_events)
    df_checkins = load_df(config.aws_bucket_name, config.s3_path_capitan_checkins)
    df_instagram = load_df(config.aws_bucket_name, config.s3_path_instagram_posts)
    df_mailchimp = load_df(config.aws_bucket_name, config.s3_path_mailchimp_campaigns)
    df_failed_payments = load_df(config.aws_bucket_name, config.s3_path_failed_payments)
    df_expenses = load_df(config.aws_bucket_name, config.s3_path_quickbooks_expenses)
    df_twilio_messages = load_df(config.aws_bucket_name, config.s3_path_twilio_messages)
    df_customer_identifiers = load_df(config.aws_bucket_name, 'customers/customer_identifiers.csv')
    df_customers_master = load_df(config.aws_bucket_name, 'customers/customers_master.csv')
    df_customer_events = load_df(config.aws_bucket_name, config.s3_path_customer_events)
    df_customer_flags = load_df(config.aws_bucket_name, config.s3_path_customer_flags)
    df_day_pass_engagement = load_df(config.aws_bucket_name, 'analytics/day_pass_engagement.csv')
    df_membership_conversion = load_df(config.aws_bucket_name, 'analytics/membership_conversion_metrics.csv')
    df_mailchimp_member_tags = load_df(config.aws_bucket_name, 'marketing/mailchimp_member_tags.csv')

    # New data sources for flag tracking and AB test analysis
    df_shopify_synced_flags = load_df(config.aws_bucket_name, config.s3_path_shopify_synced_flags)
    df_experiment_entries = load_df(config.aws_bucket_name, config.s3_path_experiment_entries)
    df_day_pass_checkin_recency = load_df(config.aws_bucket_name, config.s3_path_day_pass_checkin_recency)

    # Load Shopify orders and convert to transaction format
    df_shopify = load_df(config.aws_bucket_name, config.s3_path_shopify_orders)
    if not df_shopify.empty:
        # Convert Shopify to transaction format
        shopify_transactions = convert_shopify_to_transactions(df_shopify)
        # Merge with existing transactions
        df_transactions = pd.concat([df_transactions, shopify_transactions], ignore_index=True)

    return df_transactions, df_memberships, df_members, df_projection, df_at_risk, df_new_members, df_facebook_ads, df_events, df_checkins, df_instagram, df_mailchimp, df_failed_payments, df_expenses, df_twilio_messages, df_customer_identifiers, df_customers_master, df_customer_events, df_customer_flags, df_day_pass_engagement, df_membership_conversion, df_mailchimp_member_tags, df_shopify_synced_flags, df_experiment_entries, df_day_pass_checkin_recency


# TODO: Re-enable password protection after testing
# if not check_password():
#     st.stop()

# Load data
with st.spinner('Loading data from S3...'):
    df_transactions, df_memberships, df_members, df_projection, df_at_risk, df_new_members, df_facebook_ads, df_events, df_checkins, df_instagram, df_mailchimp, df_failed_payments, df_expenses, df_twilio_messages, df_customer_identifiers, df_customers_master, df_customer_events, df_customer_flags, df_day_pass_engagement, df_membership_conversion, df_mailchimp_member_tags, df_shopify_synced_flags, df_experiment_entries, df_day_pass_checkin_recency = load_data()

# Prepare at-risk members data
if not df_at_risk.empty:
    df_at_risk['full_name'] = df_at_risk['first_name'] + ' ' + df_at_risk['last_name']
    df_at_risk_display = df_at_risk[[
        'full_name', 'age', 'membership_type', 'last_checkin_date',
        'risk_category', 'risk_description', 'capitan_link'
    ]].copy()
    df_at_risk_display.columns = [
        'Name', 'Age', 'Membership Type', 'Last Check-in',
        'Risk Category', 'Description', 'Capitan Link'
    ]

# App title
st.title('🧗 Basin Climbing & Fitness Dashboard')
st.markdown('---')

# Create tabs
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Overview",
    "📊 Revenue",
    "👥 Membership",
    "🎟️ Day Passes & Check-ins",
    "🎉 Rentals",
    "💪 Programming",
    "📱 Marketing",
    "🎯 Lead Flow"
])

# ============================================================================
# TAB 0: OVERVIEW
# ============================================================================
with tab0:
    st.header('Overview - Key Metrics')

    # Calculate date ranges
    from datetime import datetime, timedelta
    import calendar

    today = pd.Timestamp.now()

    # Prepare data with dates
    df_transactions['Date'] = pd.to_datetime(df_transactions['Date'], errors='coerce')
    if 'checkin_datetime' in df_checkins.columns:
        df_checkins['checkin_datetime'] = pd.to_datetime(df_checkins['checkin_datetime'], errors='coerce')
    if 'created_at' in df_memberships.columns:
        df_memberships['created_at'] = pd.to_datetime(df_memberships['created_at'], errors='coerce')
    if 'event_date' in df_events.columns:
        df_events['event_date'] = pd.to_datetime(df_events['event_date'], errors='coerce')

    # ========== WEEKLY SCORECARD ==========
    st.subheader('Weekly Scorecard')

    # Calculate last 4 complete weeks (Mon-Sun)
    days_since_monday = today.weekday()
    current_monday = (today - timedelta(days=days_since_monday)).normalize()

    week_ranges = []
    for i in range(4, 0, -1):
        week_start = current_monday - timedelta(weeks=i)
        week_end = week_start + timedelta(days=6)
        week_ranges.append((week_start, week_end))

    # Prepare membership dates for scorecard
    df_mem_sc = df_memberships.copy()
    df_mem_sc['start_date'] = pd.to_datetime(df_mem_sc.get('start_date'), errors='coerce')
    df_mem_sc['end_date'] = pd.to_datetime(df_mem_sc.get('end_date'), errors='coerce')

    # Prepare birthday filter
    bday_mask = (
        (df_transactions['Description'].str.contains('Birthday Party', case=False, na=False)) |
        ((df_transactions['revenue_category'] == 'Event Booking') &
         (df_transactions['Description'].str.contains('birthday', case=False, na=False)))
    )
    if 'sub_category' in df_transactions.columns:
        bday_mask = bday_mask | (
            (df_transactions['sub_category'] == 'birthday') &
            (df_transactions['Description'].str.contains('Calendly', case=False, na=False))
        )
    df_bday = df_transactions[bday_mask].copy()

    scorecard_rows = []
    for ws, we in week_ranges:
        we_exclusive = we + timedelta(days=1)

        # Revenue
        rev = df_transactions[
            (df_transactions['Date'] >= ws) & (df_transactions['Date'] < we_exclusive)
        ]['Total Amount'].sum()

        # Day pass units
        dp = df_transactions[
            (df_transactions['Date'] >= ws) & (df_transactions['Date'] < we_exclusive) &
            (df_transactions['revenue_category'] == 'Day Pass')
        ]
        dp_units = dp['Day Pass Count'].sum() if 'Day Pass Count' in dp.columns else len(dp)

        # New memberships
        new_mem = len(df_mem_sc[
            (df_mem_sc['start_date'] >= ws) & (df_mem_sc['start_date'] < we_exclusive)
        ]) if 'start_date' in df_mem_sc.columns else 0

        # Attrited memberships
        att_mem = len(df_mem_sc[
            (df_mem_sc['status'] == 'END') &
            (df_mem_sc['end_date'] >= ws) & (df_mem_sc['end_date'] < we_exclusive)
        ]) if 'end_date' in df_mem_sc.columns and 'status' in df_mem_sc.columns else 0

        # Birthday parties
        bday_count = len(df_bday[
            (df_bday['Date'] >= ws) & (df_bday['Date'] < we_exclusive)
        ])

        # Fitness dollars
        fit = 0
        if 'fitness_amount' in df_transactions.columns:
            fit = df_transactions[
                (df_transactions['Date'] >= ws) & (df_transactions['Date'] < we_exclusive) &
                (df_transactions['fitness_amount'] > 0)
            ]['fitness_amount'].sum()

        scorecard_rows.append({
            'Week': f"{ws.strftime('%b %d')} - {we.strftime('%b %d')}",
            'Revenue': rev,
            'Day Passes': int(dp_units),
            'New Members': int(new_mem),
            'Attrition': int(att_mem),
            'Birthdays': int(bday_count),
            'Fitness $': fit
        })

    df_scorecard = pd.DataFrame(scorecard_rows)

    # Add 4-week average column
    avg_col = {
        'Week': '4-Wk Avg',
        'Revenue': df_scorecard['Revenue'].mean(),
        'Day Passes': int(round(df_scorecard['Day Passes'].mean())),
        'New Members': int(round(df_scorecard['New Members'].mean())),
        'Attrition': int(round(df_scorecard['Attrition'].mean())),
        'Birthdays': int(round(df_scorecard['Birthdays'].mean())),
        'Fitness $': df_scorecard['Fitness $'].mean()
    }
    df_scorecard = pd.concat([df_scorecard, pd.DataFrame([avg_col])], ignore_index=True)

    # Transpose: metrics as rows, weeks as columns (trends left to right)
    df_scorecard = df_scorecard.set_index('Week').T
    df_scorecard.index.name = 'Metric'

    # Format dollar rows
    for col in df_scorecard.columns:
        df_scorecard.loc['Revenue', col] = f"${df_scorecard.loc['Revenue', col]:,.0f}"
        df_scorecard.loc['Fitness $', col] = f"${df_scorecard.loc['Fitness $', col]:,.0f}"
        for metric in ['Day Passes', 'New Members', 'Attrition', 'Birthdays']:
            df_scorecard.loc[metric, col] = f"{int(df_scorecard.loc[metric, col]):,}"

    # Render as styled HTML table for larger font and centered values
    header_row = '<tr><th style="text-align:left; padding:10px 16px; font-size:18px; border-bottom:2px solid #213B3F;">Metric</th>'
    for col in df_scorecard.columns:
        header_row += f'<th style="text-align:center; padding:10px 16px; font-size:18px; border-bottom:2px solid #213B3F;">{col}</th>'
    header_row += '</tr>'

    body_rows = ''
    for metric in df_scorecard.index:
        body_rows += f'<tr><td style="text-align:left; padding:8px 16px; font-size:17px; font-weight:600; border-bottom:1px solid #E0E0E0;">{metric}</td>'
        for col in df_scorecard.columns:
            body_rows += f'<td style="text-align:center; padding:8px 16px; font-size:17px; border-bottom:1px solid #E0E0E0;">{df_scorecard.loc[metric, col]}</td>'
        body_rows += '</tr>'

    scorecard_html = f'''
    <table style="width:100%; border-collapse:collapse; background-color:#FFFFFF; border-radius:8px;">
        <thead>{header_row}</thead>
        <tbody>{body_rows}</tbody>
    </table>
    '''
    st.markdown(scorecard_html, unsafe_allow_html=True)

# ============================================================================
# TAB 1: REVENUE
# ============================================================================
with tab1:
    st.header('Revenue Analysis')

    # Timeframe selector
    timeframe = st.selectbox(
        'Select Timeframe',
        options=['D', 'W', 'M'],
        format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[x],
        index=2  # Default to Monthly
    )

    # Data source selector
    data_sources = st.multiselect(
        'Data Sources',
        options=['Stripe', 'Square', 'Shopify'],
        default=['Stripe', 'Square', 'Shopify']
    )

    # Filter data
    df_filtered = df_transactions[df_transactions['Data Source'].isin(data_sources)].copy()
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')
    df_filtered = df_filtered[df_filtered['Date'].notna()]
    df_filtered['date'] = df_filtered['Date'].dt.to_period(timeframe).dt.start_time

    # Revenue by category
    revenue_by_category = (
        df_filtered.groupby(['date', 'revenue_category'])['Total Amount']
        .sum()
        .reset_index()
    )

    category_order = [
        'Day Pass', 'New Membership', 'Membership Renewal',
        'Programming', 'Team Dues', 'Retail', 'Event Booking'
    ]

    # Line chart - Total Revenue Over Time
    st.subheader('Total Revenue Over Time')
    total_revenue = df_filtered.groupby('date')['Total Amount'].sum().reset_index()

    fig_line = px.line(
        total_revenue,
        x='date',
        y='Total Amount',
        title='Total Revenue Over Time',
        text=total_revenue['Total Amount'].apply(lambda x: f'${x/1000:.1f}K')
    )
    fig_line.update_traces(
        line_color=COLORS['primary'],
        textposition='top center',
        textfont=dict(size=14)
    )
    fig_line.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text'],
        yaxis_title='Revenue ($)',
        xaxis_title='Date',
        xaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        )
    )
    fig_line = apply_axis_styling(fig_line)
    st.plotly_chart(fig_line, use_container_width=True)

    # Stacked bar chart - Revenue by Category
    st.subheader('Revenue by Category (Stacked)')
    fig_stacked = px.bar(
        revenue_by_category,
        x='date',
        y='Total Amount',
        color='revenue_category',
        title='Revenue by Category',
        barmode='stack',
        category_orders={'revenue_category': category_order},
        color_discrete_map=REVENUE_CATEGORY_COLORS
    )
    fig_stacked.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text'],
        yaxis_title='Revenue ($)',
        xaxis_title='Date',
        legend_title='Category',
        xaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        )
    )
    fig_stacked = apply_axis_styling(fig_stacked)
    st.plotly_chart(fig_stacked, use_container_width=True)

    # Percentage chart - Revenue by Category
    st.subheader('Percentage of Revenue by Category')
    total_revenue_per_date = (
        revenue_by_category.groupby('date')['Total Amount'].sum().reset_index()
    )
    total_revenue_per_date.columns = ['date', 'total_revenue']
    revenue_with_total = pd.merge(revenue_by_category, total_revenue_per_date, on='date')
    revenue_with_total['percentage'] = (
        revenue_with_total['Total Amount'] / revenue_with_total['total_revenue']
    ) * 100

    fig_percentage = px.bar(
        revenue_with_total,
        x='date',
        y='percentage',
        color='revenue_category',
        title='Percentage of Revenue by Category',
        barmode='stack',
        category_orders={'revenue_category': category_order},
        text=revenue_with_total['percentage'].apply(lambda x: f'{x:.1f}%'),
        color_discrete_map=REVENUE_CATEGORY_COLORS
    )
    fig_percentage.update_traces(
        textposition='outside',  # Always position labels outside for consistent size/visibility
        textfont=dict(size=14),
        cliponaxis=False
    )
    fig_percentage.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text'],
        yaxis_title='Percentage (%)',
        xaxis_title='Date',
        legend_title='Category',
        xaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        )
    )
    fig_percentage = apply_axis_styling(fig_percentage)
    st.plotly_chart(fig_percentage, use_container_width=True)

    # Refund rate chart
    st.subheader('Refund Rate by Category')
    df_filtered_copy = df_filtered.copy()
    df_filtered_copy['is_refund'] = df_filtered_copy['Total Amount'] < 0

    refund_stats = df_filtered_copy.groupby('revenue_category').agg({
        'Total Amount': lambda x: {
            'gross': x[x > 0].sum(),
            'refunds': abs(x[x < 0].sum()),
            'net': x.sum()
        }
    }).reset_index()

    refund_stats['gross_revenue'] = refund_stats['Total Amount'].apply(lambda x: x['gross'])
    refund_stats['refunds'] = refund_stats['Total Amount'].apply(lambda x: x['refunds'])
    refund_stats['net_revenue'] = refund_stats['Total Amount'].apply(lambda x: x['net'])
    refund_stats.drop('Total Amount', axis=1, inplace=True)

    refund_stats['refund_rate'] = (refund_stats['refunds'] / refund_stats['gross_revenue'] * 100).fillna(0)
    refund_stats = refund_stats[refund_stats['gross_revenue'] > 0]
    refund_stats = refund_stats.sort_values('refund_rate', ascending=False)

    fig_refund = px.bar(
        refund_stats,
        y='revenue_category',
        x='refund_rate',
        title='Refund Rate by Category (%)',
        orientation='h',
        text='refund_rate'
    )
    fig_refund.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        marker_color=COLORS['primary']
    )
    fig_refund.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text'],
        xaxis_title='Refund Rate (%)',
        yaxis_title='Category',
        height=400,
        xaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        )
    )
    fig_refund = apply_axis_styling(fig_refund)
    st.plotly_chart(fig_refund, use_container_width=True)

    # Accounting groups chart
    st.subheader('Revenue by Accounting Groups')
    accounting_groups = revenue_by_category.copy()

    def map_to_accounting_group(category):
        if category in ['New Membership', 'Membership Renewal']:
            return 'Memberships'
        elif category in ['Team Dues', 'Programming']:
            return 'Team & Programming'
        else:
            return category

    accounting_groups['accounting_group'] = accounting_groups['revenue_category'].apply(map_to_accounting_group)
    accounting_revenue = accounting_groups.groupby(['date', 'accounting_group'])['Total Amount'].sum().reset_index()

    accounting_total = accounting_revenue.groupby('date')['Total Amount'].sum().reset_index()
    accounting_total.columns = ['date', 'total_revenue']
    accounting_with_total = pd.merge(accounting_revenue, accounting_total, on='date')
    accounting_with_total['percentage'] = (accounting_with_total['Total Amount'] / accounting_with_total['total_revenue']) * 100

    accounting_colors = {
        'Memberships': COLORS['primary'],         # Rust
        'Team & Programming': COLORS['tertiary'], # Sage green
        'Day Pass': COLORS['quaternary'],         # Dark teal
        'Retail': '#8B7355',                      # Brown (distinct from memberships)
        'Event Booking': '#B8C9A8',               # Light sage
    }

    fig_accounting = px.bar(
        accounting_with_total,
        x='date',
        y='percentage',
        color='accounting_group',
        title='Revenue by Accounting Groups (Memberships, Team & Programming, etc.)',
        barmode='stack',
        text=accounting_with_total['percentage'].apply(lambda x: f'{x:.1f}%'),
        color_discrete_map=accounting_colors
    )
    fig_accounting.update_traces(
        textposition='auto',  # Auto positions text inside when fits, outside when too small
        textfont=dict(size=14, color='white'),
        insidetextanchor='middle'
    )
    fig_accounting.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text'],
        yaxis_title='Percentage (%)',
        xaxis_title='Date',
        legend_title='Group',
        xaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        )
    )
    fig_accounting = apply_axis_styling(fig_accounting)
    st.plotly_chart(fig_accounting, use_container_width=True)

    # Membership Revenue Projection
    st.subheader('Membership Revenue: Historical & Projected')

    # Get historical membership revenue (last 3 months)
    df_membership_revenue = df_transactions[
        df_transactions['revenue_category'].isin(['New Membership', 'Membership Renewal'])
    ].copy()
    df_membership_revenue['date'] = pd.to_datetime(df_membership_revenue['Date'])
    df_membership_revenue['month'] = df_membership_revenue['date'].dt.to_period('M')

    historical_revenue = (
        df_membership_revenue.groupby('month')['Total Amount']
        .sum()
        .reset_index()
    )
    historical_revenue.columns = ['month', 'amount']
    historical_revenue['month_str'] = historical_revenue['month'].astype(str)
    historical_revenue['type'] = 'Realized'

    # Get current month
    current_month = pd.Period.now('M')

    # Filter to last 3 months
    historical_revenue = historical_revenue[
        historical_revenue['month'] >= (current_month - 2)
    ]

    # Get projected revenue
    df_proj = df_projection.copy()
    df_proj['date'] = pd.to_datetime(df_proj['date'])
    df_proj['month'] = df_proj['date'].dt.to_period('M')

    proj_summary = df_proj.groupby('month')['projected_total'].sum().reset_index()
    proj_summary.columns = ['month', 'amount']
    proj_summary['month_str'] = proj_summary['month'].astype(str)
    proj_summary['type'] = 'Projected'

    # Filter to next 4 months (including current month)
    proj_summary = proj_summary[
        (proj_summary['month'] >= current_month) &
        (proj_summary['month'] <= current_month + 3)
    ]

    # For current month, we want both realized (so far) and projected (scheduled)
    # Get realized revenue for current month
    current_month_realized = historical_revenue[
        historical_revenue['month'] == current_month
    ].copy()

    # Remove current month from projected (we'll add it back with both components)
    proj_summary_future = proj_summary[proj_summary['month'] > current_month].copy()
    current_month_projected = proj_summary[proj_summary['month'] == current_month].copy()

    # Combine data for chart
    # Past months: realized only
    past_months = historical_revenue[historical_revenue['month'] < current_month].copy()

    # Current month: both realized and projected stacked
    # Future months: projected only

    # Create chart data
    chart_data = []

    # Add past months (realized)
    for _, row in past_months.iterrows():
        chart_data.append({
            'month': row['month_str'],
            'Realized': row['amount'],
            'Projected': 0
        })

    # Add current month (both)
    if not current_month_realized.empty and not current_month_projected.empty:
        chart_data.append({
            'month': current_month.strftime('%Y-%m'),
            'Realized': current_month_realized['amount'].iloc[0],
            'Projected': current_month_projected['amount'].iloc[0]
        })
    elif not current_month_projected.empty:
        # No realized revenue yet this month
        chart_data.append({
            'month': current_month.strftime('%Y-%m'),
            'Realized': 0,
            'Projected': current_month_projected['amount'].iloc[0]
        })

    # Add future months (projected only)
    for _, row in proj_summary_future.iterrows():
        chart_data.append({
            'month': row['month_str'],
            'Realized': 0,
            'Projected': row['amount']
        })

    df_chart = pd.DataFrame(chart_data)

    # Create stacked bar chart
    fig_projection = go.Figure()

    # Add realized revenue bars
    fig_projection.add_trace(go.Bar(
        name='Realized',
        x=df_chart['month'],
        y=df_chart['Realized'],
        marker_color=COLORS['primary'],
        text=df_chart['Realized'].apply(lambda x: f'${x/1000:.1f}K' if x > 0 else ''),
        textposition='inside',
        textfont=dict(size=14, color='white')
    ))

    # Add projected revenue bars
    fig_projection.add_trace(go.Bar(
        name='Projected',
        x=df_chart['month'],
        y=df_chart['Projected'],
        marker_color=COLORS['secondary'],
        text=df_chart['Projected'].apply(lambda x: f'${x/1000:.1f}K' if x > 0 else ''),
        textposition='inside',
        textfont=dict(size=14, color='white')
    ))

    fig_projection.update_layout(
        barmode='stack',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text'],
        yaxis_title='Revenue ($)',
        xaxis_title='Month',
        legend_title='Type',
        showlegend=True,
        xaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        )
    )

    fig_projection = apply_axis_styling(fig_projection)
    st.plotly_chart(fig_projection, use_container_width=True)

    # Payment Failure Rates
    st.subheader('Payment Failure Rates by Membership Type')
    st.markdown('Analysis of failed membership payments over the last 180 days')

    if not df_failed_payments.empty and not df_memberships.empty:
        from data_pipeline.process_failed_payments import calculate_failure_rates_by_type

        # Calculate failure rates
        df_failure_rates = calculate_failure_rates_by_type(df_failed_payments, df_memberships)

        # Filter to show only categories with >0% failure rate
        df_failure_rates_display = df_failure_rates[df_failure_rates['failure_rate_pct'] > 0].copy()

        if not df_failure_rates_display.empty:
            # Create two columns for metrics
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Total Failed Payments",
                    len(df_failed_payments),
                    help="Failed membership payment attempts in last 180 days"
                )

            with col2:
                insufficient_funds_count = len(df_failed_payments[df_failed_payments['decline_code'] == 'insufficient_funds'])
                insufficient_funds_pct = (insufficient_funds_count / len(df_failed_payments) * 100) if len(df_failed_payments) > 0 else 0
                st.metric(
                    "Due to Insufficient Funds",
                    f"{insufficient_funds_count} ({insufficient_funds_pct:.1f}%)",
                    help="Failures specifically due to insufficient funds"
                )

            # Create bar chart showing failure rates
            fig_failures = go.Figure()

            # Sort by insufficient funds rate descending
            df_failure_rates_display = df_failure_rates_display.sort_values('insufficient_funds_rate_pct', ascending=True)

            # Create hover text with detailed info
            df_failure_rates_display['hover_text'] = df_failure_rates_display.apply(
                lambda row: (
                    f"<b>{row['membership_type']}</b><br>" +
                    f"Active Members: {row['active_memberships']}<br>" +
                    f"Failed Payments: {row['total_failures']}<br>" +
                    f"Insufficient Funds: {row['insufficient_funds_failures']}<br>" +
                    f"Failure Rate: {row['failure_rate_pct']:.1f}%"
                ),
                axis=1
            )

            # Add insufficient funds rate
            fig_failures.add_trace(go.Bar(
                name='Insufficient Funds',
                y=df_failure_rates_display['membership_type'],
                x=df_failure_rates_display['insufficient_funds_rate_pct'],
                orientation='h',
                marker_color=COLORS['primary'],
                text=df_failure_rates_display.apply(
                    lambda row: f'{row["insufficient_funds_rate_pct"]:.1f}% ({row["insufficient_funds_failures"]} fails)',
                    axis=1
                ),
                textposition='auto',
                hovertext=df_failure_rates_display['hover_text'],
                hoverinfo='text',
            ))

            # Add other failures rate
            df_failure_rates_display['other_failure_rate'] = (
                df_failure_rates_display['failure_rate_pct'] - df_failure_rates_display['insufficient_funds_rate_pct']
            )
            df_failure_rates_display['other_failures_count'] = (
                df_failure_rates_display['total_failures'] - df_failure_rates_display['insufficient_funds_failures']
            )

            fig_failures.add_trace(go.Bar(
                name='Other Failures',
                y=df_failure_rates_display['membership_type'],
                x=df_failure_rates_display['other_failure_rate'],
                orientation='h',
                marker_color=COLORS['secondary'],
                text=df_failure_rates_display.apply(
                    lambda row: f'{row["other_failure_rate"]:.1f}% ({row["other_failures_count"]} fails)' if row['other_failure_rate'] > 0 else '',
                    axis=1
                ),
                textposition='auto',
                hovertext=df_failure_rates_display['hover_text'],
                hoverinfo='text',
            ))

            fig_failures.update_layout(
                barmode='stack',
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['background'],
                font_color=COLORS['text'],
                xaxis_title='Failure Rate (%)',
                yaxis_title='Membership Type',
                legend_title='Failure Reason',
                showlegend=True,
                height=400
            )

            fig_failures = apply_axis_styling(fig_failures)
            st.plotly_chart(fig_failures, use_container_width=True)

            # Show detailed table
            with st.expander("📊 View Detailed Failure Rates"):
                df_failure_rates_table = df_failure_rates_display[[
                    'membership_type', 'active_memberships', 'unique_with_failures',
                    'insufficient_funds_failures', 'failure_rate_pct', 'insufficient_funds_rate_pct'
                ]].copy()

                df_failure_rates_table.columns = [
                    'Membership Type', 'Active Members', 'Members with Failures',
                    'Insufficient Funds Count', 'Total Failure Rate (%)', 'Insufficient Funds Rate (%)'
                ]

                st.dataframe(df_failure_rates_table, use_container_width=True, hide_index=True)

                # Key insights
                st.markdown("**Key Insights:**")
                highest_insuff = df_failure_rates_display.iloc[-1]  # Last row (highest after sorting)
                st.markdown(f"- **{highest_insuff['membership_type']}** has the highest insufficient funds rate at **{highest_insuff['insufficient_funds_rate_pct']:.1f}%**")

                total_unique_failures = df_failure_rates_display['unique_with_failures'].sum()
                st.markdown(f"- **{total_unique_failures}** unique memberships have experienced payment failures")

                if insufficient_funds_count > 0:
                    st.markdown(f"- **{insufficient_funds_pct:.1f}%** of all payment failures are due to insufficient funds")

        else:
            st.info("No payment failures in the last 180 days!")

    else:
        st.info("Payment failure data not available")

# ============================================================================
# TAB 2: MEMBERSHIP
# ============================================================================
with tab2:
    st.header('Membership Analysis')

    # Membership Timeline
    st.subheader('Active Memberships Over Time')

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.multiselect(
            'Status',
            options=['ACT', 'END', 'FRZ'],
            default=['ACT', 'END'],
            format_func=lambda x: {'ACT': 'Active', 'END': 'Ended', 'FRZ': 'Frozen'}[x]
        )

    with col2:
        frequency_filter = st.multiselect(
            'Frequency',
            options=['bi_weekly', 'monthly', 'annual', 'prepaid_3mo', 'prepaid_6mo', 'prepaid_12mo'],
            default=['bi_weekly', 'monthly', 'annual', 'prepaid_3mo', 'prepaid_6mo', 'prepaid_12mo'],
            format_func=lambda x: x.replace('_', ' ').title()
        )

    with col3:
        size_filter = st.multiselect(
            'Size',
            options=['solo', 'duo', 'family', 'corporate'],
            default=['solo', 'duo', 'family', 'corporate'],
            format_func=lambda x: x.title()
        )

    # Category filters
    category_options = {
        'founder': 'Founder',
        'college': 'College',
        'corporate': 'Corporate',
        'mid_day': 'Mid-Day',
        'fitness_only': 'Fitness Only',
        'has_fitness_addon': 'Has Fitness Addon',
        'team_dues': 'Team Dues',
        '90_for_90': '90 for 90',
        'not_special': 'Not in Special Category'
    }

    category_filter = st.multiselect(
        'Special Categories',
        options=list(category_options.keys()),
        default=['founder', 'college', 'corporate', 'mid_day', 'fitness_only', 'has_fitness_addon', 'team_dues', '90_for_90', 'not_special'],
        format_func=lambda x: category_options[x]
    )

    # Filter memberships - Status, Frequency, Size
    df_memberships_filtered = df_memberships[df_memberships['status'].isin(status_filter)].copy()

    # Only apply frequency filter if something is selected
    if frequency_filter:
        df_memberships_filtered = df_memberships_filtered[df_memberships_filtered['frequency'].isin(frequency_filter)]

    # Only apply size filter if something is selected
    if size_filter:
        df_memberships_filtered = df_memberships_filtered[df_memberships_filtered['size'].isin(size_filter)]

    # Apply category filters - only exclude if NOT selected
    if 'founder' not in category_filter:
        df_memberships_filtered = df_memberships_filtered[~df_memberships_filtered['is_founder']]
    if 'college' not in category_filter:
        df_memberships_filtered = df_memberships_filtered[~df_memberships_filtered['is_college']]
    if 'corporate' not in category_filter:
        df_memberships_filtered = df_memberships_filtered[~df_memberships_filtered['is_corporate']]
    if 'mid_day' not in category_filter:
        df_memberships_filtered = df_memberships_filtered[~df_memberships_filtered['is_mid_day']]
    if 'fitness_only' not in category_filter:
        df_memberships_filtered = df_memberships_filtered[~df_memberships_filtered['is_fitness_only']]
    if 'has_fitness_addon' not in category_filter:
        df_memberships_filtered = df_memberships_filtered[~df_memberships_filtered['has_fitness_addon']]
    if 'team_dues' not in category_filter:
        df_memberships_filtered = df_memberships_filtered[~df_memberships_filtered['is_team_dues']]
    if '90_for_90' not in category_filter:
        df_memberships_filtered = df_memberships_filtered[~df_memberships_filtered['is_90_for_90']]
    if 'not_special' not in category_filter:
        # Exclude members NOT in any special category (i.e., only show special category members)
        special_mask = (
            df_memberships_filtered['is_founder'] |
            df_memberships_filtered['is_college'] |
            df_memberships_filtered['is_corporate'] |
            df_memberships_filtered['is_mid_day'] |
            df_memberships_filtered['is_fitness_only'] |
            df_memberships_filtered['has_fitness_addon'] |
            df_memberships_filtered['is_team_dues'] |
            df_memberships_filtered['is_90_for_90']
        )
        df_memberships_filtered = df_memberships_filtered[special_mask]

    # Process dates
    df_memberships_filtered['start_date'] = pd.to_datetime(df_memberships_filtered['start_date'], errors='coerce')
    df_memberships_filtered['end_date'] = pd.to_datetime(df_memberships_filtered['end_date'], errors='coerce')

    if not df_memberships_filtered.empty:
        min_date = df_memberships_filtered['start_date'].min()
        max_date = pd.Timestamp.now()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        daily_counts = []
        for date in date_range:
            active = df_memberships_filtered[
                (df_memberships_filtered['start_date'] <= date) &
                (df_memberships_filtered['end_date'] >= date)
            ]
            counts = active['frequency'].value_counts().to_dict()
            daily_counts.append({
                'date': date,
                **{freq: counts.get(freq, 0) for freq in frequency_filter}
            })

        daily_counts_df = pd.DataFrame(daily_counts)

        # Create stacked area chart
        fig_timeline = go.Figure()

        frequency_colors = {
            'bi_weekly': '#1f77b4',
            'monthly': '#ff7f0e',
            'annual': '#2ca02c',
            'prepaid_3mo': '#8B4229',
            'prepaid_6mo': '#BAA052',
            'prepaid_12mo': '#96A682',
        }

        for freq in frequency_filter:
            if freq in daily_counts_df.columns:
                fig_timeline.add_trace(go.Scatter(
                    x=daily_counts_df['date'],
                    y=daily_counts_df[freq],
                    mode='lines',
                    name=freq.replace('_', ' ').title(),
                    stackgroup='one',
                    line=dict(color=frequency_colors.get(freq, COLORS['primary']))
                ))

        # Add total line
        total = daily_counts_df[frequency_filter].sum(axis=1)
        fig_timeline.add_trace(go.Scatter(
            x=daily_counts_df['date'],
            y=total,
            mode='lines',
            name='Total',
            line=dict(color='#222222', width=2, dash='dash')
        ))

        fig_timeline.update_layout(
            title='Active Memberships Over Time by Payment Frequency',
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            height=600,
            xaxis_title='Date',
            yaxis_title='Number of Active Memberships',
            hovermode='x unified'
        )

        fig_timeline = apply_axis_styling(fig_timeline)
        st.plotly_chart(fig_timeline, use_container_width=True)

    # Active Members Over Time
    st.subheader('Active Members Over Time')

    # Use same filters as above but work with df_members
    df_members_filtered = df_members[df_members['status'].isin(status_filter)].copy()

    # Only apply frequency filter if something is selected
    if frequency_filter:
        df_members_filtered = df_members_filtered[df_members_filtered['frequency'].isin(frequency_filter)]

    # Only apply size filter if something is selected
    if size_filter:
        df_members_filtered = df_members_filtered[df_members_filtered['size'].isin(size_filter)]

    # Apply category filters - only exclude if NOT selected
    if 'founder' not in category_filter:
        df_members_filtered = df_members_filtered[~df_members_filtered['is_founder']]
    if 'college' not in category_filter:
        df_members_filtered = df_members_filtered[~df_members_filtered['is_college']]
    if 'corporate' not in category_filter:
        df_members_filtered = df_members_filtered[~df_members_filtered['is_corporate']]
    if 'mid_day' not in category_filter:
        df_members_filtered = df_members_filtered[~df_members_filtered['is_mid_day']]
    if 'fitness_only' not in category_filter:
        df_members_filtered = df_members_filtered[~df_members_filtered['is_fitness_only']]
    if 'has_fitness_addon' not in category_filter:
        df_members_filtered = df_members_filtered[~df_members_filtered['has_fitness_addon']]
    if 'team_dues' not in category_filter:
        df_members_filtered = df_members_filtered[~df_members_filtered['is_team_dues']]
    if '90_for_90' not in category_filter:
        df_members_filtered = df_members_filtered[~df_members_filtered['is_90_for_90']]
    if 'not_special' not in category_filter:
        # Exclude members NOT in any special category (i.e., only show special category members)
        special_mask = (
            df_members_filtered['is_founder'] |
            df_members_filtered['is_college'] |
            df_members_filtered['is_corporate'] |
            df_members_filtered['is_mid_day'] |
            df_members_filtered['is_fitness_only'] |
            df_members_filtered['has_fitness_addon'] |
            df_members_filtered['is_team_dues'] |
            df_members_filtered['is_90_for_90']
        )
        df_members_filtered = df_members_filtered[special_mask]

    # Process dates
    df_members_filtered['start_date'] = pd.to_datetime(df_members_filtered['start_date'], errors='coerce')
    df_members_filtered['end_date'] = pd.to_datetime(df_members_filtered['end_date'], errors='coerce')

    if not df_members_filtered.empty:
        min_date = df_members_filtered['start_date'].min()
        max_date = pd.Timestamp.now()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        daily_member_counts = []
        for date in date_range:
            active_members = df_members_filtered[
                (df_members_filtered['start_date'] <= date) &
                (df_members_filtered['end_date'] >= date)
            ]
            count = len(active_members)
            daily_member_counts.append({
                'date': date,
                'count': count
            })

        daily_members_df = pd.DataFrame(daily_member_counts)

        # Create line chart
        fig_members_timeline = px.line(
            daily_members_df,
            x='date',
            y='count',
            title='Active Individual Members Over Time',
            line_shape='linear'
        )
        fig_members_timeline.update_traces(line_color=COLORS['primary'], line_width=2)
        fig_members_timeline.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            height=500,
            xaxis_title='Date',
            yaxis_title='Number of Active Members',
            hovermode='x unified'
        )

        fig_members_timeline = apply_axis_styling(fig_members_timeline)
        st.plotly_chart(fig_members_timeline, use_container_width=True)
    else:
        st.info('No members match the selected filters')

    # Memberships by Size
    st.subheader('Active Memberships by Group Size')

    if not df_memberships_filtered.empty:
        daily_size_counts = []
        for date in date_range:
            active = df_memberships_filtered[
                (df_memberships_filtered['start_date'] <= date) &
                (df_memberships_filtered['end_date'] >= date)
            ]
            counts = active['size'].value_counts().to_dict()
            daily_size_counts.append({
                'date': date,
                **{size: counts.get(size, 0) for size in size_filter}
            })

        daily_size_df = pd.DataFrame(daily_size_counts)

        # Create stacked area chart
        fig_size = go.Figure()

        size_colors = {
            'solo': COLORS['primary'],
            'duo': COLORS['secondary'],
            'family': COLORS['tertiary'],
            'corporate': COLORS['quaternary']
        }

        for size in size_filter:
            if size in daily_size_df.columns:
                fig_size.add_trace(go.Scatter(
                    x=daily_size_df['date'],
                    y=daily_size_df[size],
                    mode='lines',
                    name=size.title(),
                    stackgroup='one',
                    line=dict(color=size_colors.get(size, COLORS['primary']))
                ))

        # Add total line
        total_size = daily_size_df[size_filter].sum(axis=1)
        fig_size.add_trace(go.Scatter(
            x=daily_size_df['date'],
            y=total_size,
            mode='lines',
            name='Total',
            line=dict(color='#222222', width=2, dash='dash')
        ))

        fig_size.update_layout(
            title='Active Memberships Over Time by Group Size',
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            height=600,
            xaxis_title='Date',
            yaxis_title='Number of Active Memberships',
            hovermode='x unified'
        )

        fig_size = apply_axis_styling(fig_size)
        st.plotly_chart(fig_size, use_container_width=True)
    else:
        st.info('No memberships match the selected filters')

    # Memberships by Special Category
    st.subheader('Active Memberships by Special Category')

    if not df_memberships_filtered.empty:
        # Define categories to track
        special_categories = {
            'Founder': 'is_founder',
            'College': 'is_college',
            'Corporate': 'is_corporate',
            'Mid-Day': 'is_mid_day',
            'Fitness Only': 'is_fitness_only',
            'Has Fitness Addon': 'has_fitness_addon',
            'Team Dues': 'is_team_dues',
            '90 for 90': 'is_90_for_90',
            'Regular': 'regular'  # Not in any special category
        }

        daily_category_counts = []
        for date in date_range:
            active = df_memberships_filtered[
                (df_memberships_filtered['start_date'] <= date) &
                (df_memberships_filtered['end_date'] >= date)
            ]

            counts = {}
            for category_name, column_name in special_categories.items():
                if column_name == 'regular':
                    # Count memberships NOT in any special category
                    regular_mask = ~(
                        active['is_founder'] |
                        active['is_college'] |
                        active['is_corporate'] |
                        active['is_mid_day'] |
                        active['is_fitness_only'] |
                        active['has_fitness_addon'] |
                        active['is_team_dues'] |
                        active['is_90_for_90']
                    )
                    counts[category_name] = regular_mask.sum()
                else:
                    counts[category_name] = active[column_name].sum()

            daily_category_counts.append({
                'date': date,
                **counts
            })

        daily_category_df = pd.DataFrame(daily_category_counts)

        # Create stacked area chart
        fig_category = go.Figure()

        category_colors = {
            'Founder': '#8B4229',
            'College': '#BAA052',
            'Corporate': '#96A682',
            'Mid-Day': '#1A2E31',
            'Fitness Only': '#C85A3E',
            'Has Fitness Addon': '#D4AF6A',
            'Team Dues': '#B8C9A8',
            '90 for 90': '#4A4A4A',
            'Regular': '#E0E0E0'
        }

        for category in special_categories.keys():
            if category in daily_category_df.columns:
                fig_category.add_trace(go.Scatter(
                    x=daily_category_df['date'],
                    y=daily_category_df[category],
                    mode='lines',
                    name=category,
                    stackgroup='one',
                    line=dict(color=category_colors.get(category, COLORS['primary']))
                ))

        # Add total line
        total_category = daily_category_df[list(special_categories.keys())].sum(axis=1)
        fig_category.add_trace(go.Scatter(
            x=daily_category_df['date'],
            y=total_category,
            mode='lines',
            name='Total',
            line=dict(color='#222222', width=2, dash='dash')
        ))

        fig_category.update_layout(
            title='Active Memberships Over Time by Special Category',
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            height=600,
            xaxis_title='Date',
            yaxis_title='Number of Active Memberships',
            hovermode='x unified'
        )

        fig_category = apply_axis_styling(fig_category)
        st.plotly_chart(fig_category, use_container_width=True)
    else:
        st.info('No memberships match the selected filters')

    # 90 for 90 Conversion
    st.subheader('90 for 90 Conversion Summary')

    ninety_members = df_members[df_members['is_90_for_90'] == True].copy()

    if not ninety_members.empty:
        ninety_members['person_id'] = ninety_members['member_first_name'] + ' ' + ninety_members['member_last_name']
        df_members_copy = df_members.copy()
        df_members_copy['person_id'] = df_members_copy['member_first_name'] + ' ' + df_members_copy['member_last_name']

        unique_person_ids = ninety_members['person_id'].unique()

        converted_count = 0
        not_converted_count = 0

        for person_id in unique_person_ids:
            person_ninety = ninety_members[ninety_members['person_id'] == person_id]
            ninety_start_date = pd.to_datetime(person_ninety['start_date'].min(), errors='coerce')

            if pd.notna(ninety_start_date):
                regular_memberships = df_members_copy[
                    (df_members_copy['person_id'] == person_id) &
                    (df_members_copy['is_90_for_90'] == False) &
                    (pd.to_datetime(df_members_copy['start_date'], errors='coerce') > ninety_start_date)
                ]
                if len(regular_memberships) > 0:
                    converted_count += 1
                else:
                    not_converted_count += 1

        total = converted_count + not_converted_count
        conversion_rate = (converted_count / total * 100) if total > 0 else 0

        summary_data = pd.DataFrame({
            'Status': ['Converted', 'Not Converted'],
            'Count': [converted_count, not_converted_count]
        })

        fig_90 = px.bar(
            summary_data,
            x='Status',
            y='Count',
            title=f'90 for 90 Conversion Summary (Conversion Rate: {conversion_rate:.1f}%)',
            color='Status',
            color_discrete_map={
                'Converted': COLORS['secondary'],
                'Not Converted': COLORS['primary']
            }
        )
        fig_90.update_traces(texttemplate='%{y}', textposition='outside')
        fig_90.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            height=400,
            showlegend=False,
            yaxis_title='Number of Members'
        )
        fig_90 = apply_axis_styling(fig_90)
        st.plotly_chart(fig_90, use_container_width=True)
    else:
        st.info('No 90 for 90 memberships found')

    # New Members & Attrition
    st.subheader('New Memberships & Attrition Over Time')

    attrition_period = st.radio(
        'View by:',
        ['Monthly', 'Weekly'],
        horizontal=True,
        key='attrition_period_toggle'
    )

    # Calculate new memberships and attrition - use FULL dataframe (not filtered)
    # This ensures numbers match the Overview tab
    df_memberships_dates = df_memberships.copy()

    # Process dates
    df_memberships_dates['start_date'] = pd.to_datetime(df_memberships_dates['start_date'], errors='coerce')
    df_memberships_dates['end_date'] = pd.to_datetime(df_memberships_dates['end_date'], errors='coerce')

    # Remove rows with invalid dates
    df_memberships_dates = df_memberships_dates[df_memberships_dates['start_date'].notna()]

    # Get date range
    min_date = df_memberships_dates['start_date'].min()
    max_date = pd.Timestamp.now()

    if attrition_period == 'Monthly':
        period_range = pd.period_range(start=min_date.to_period('M'), end=max_date.to_period('M'), freq='M')
    else:
        period_range = pd.period_range(start=min_date.to_period('W'), end=max_date.to_period('W'), freq='W')

    period_data = []
    for period in period_range:
        period_start = period.to_timestamp()
        period_end = (period + 1).to_timestamp()

        # Count new memberships that started in this period
        new_members = len(df_memberships_dates[
            (df_memberships_dates['start_date'] >= period_start) &
            (df_memberships_dates['start_date'] < period_end)
        ])

        # Count memberships that ended in this period (attrition)
        # ONLY count memberships with status='END' to avoid counting active memberships' billing dates
        attrited = len(df_memberships_dates[
            (df_memberships_dates['status'] == 'END') &
            (df_memberships_dates['end_date'] >= period_start) &
            (df_memberships_dates['end_date'] < period_end)
        ])

        # Net change
        net_change = new_members - attrited

        period_data.append({
            'period': period.to_timestamp(),
            'New Memberships': new_members,
            'Attrition': attrited,
            'Net Change': net_change
        })

    df_period = pd.DataFrame(period_data)

    if not df_period.empty:
        # Create figure with secondary y-axis
        fig_attrition = go.Figure()

        # Add new memberships bars
        fig_attrition.add_trace(go.Bar(
            x=df_period['period'],
            y=df_period['New Memberships'],
            name='New Memberships',
            marker_color=COLORS['secondary'],
            text=df_period['New Memberships'],
            textposition='outside',
            textfont=dict(size=14)
        ))

        # Add attrition bars (negative values for visual effect)
        fig_attrition.add_trace(go.Bar(
            x=df_period['period'],
            y=-df_period['Attrition'],  # Negative to show below axis
            name='Attrition',
            marker_color=COLORS['primary'],
            text=df_period['Attrition'],
            textposition='outside',
            textfont=dict(size=14)
        ))

        # Add net change line
        fig_attrition.add_trace(go.Scatter(
            x=df_period['period'],
            y=df_period['Net Change'],
            name='Net Change',
            mode='lines+markers',
            line=dict(color=COLORS['quaternary'], width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))

        period_label = 'Month' if attrition_period == 'Monthly' else 'Week'
        fig_attrition.update_layout(
            title=f'{attrition_period} New Memberships & Attrition',
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            height=500,
            xaxis_title=period_label,
            yaxis_title='Count',
            yaxis2=dict(
                title='Net Change',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            hovermode='x unified',
            barmode='relative',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )

        # Add horizontal line at y=0
        fig_attrition.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)

        fig_attrition = apply_axis_styling(fig_attrition)
        st.plotly_chart(fig_attrition, use_container_width=True)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total New Memberships', df_period['New Memberships'].sum())
        with col2:
            st.metric('Total Attrition', df_period['Attrition'].sum())
        with col3:
            net_total = df_period['Net Change'].sum()
            st.metric('Net Growth', net_total, delta=None)
    else:
        st.info('No membership data available for attrition analysis')

    # New vs Existing Memberships Chart
    st.subheader('Active Memberships: New vs Existing')
    st.markdown('Shows the composition of active memberships: new members (joined that month) vs existing members')

    # Calculate active memberships split by new vs existing for each month
    if not df_memberships_dates.empty:
        month_range = pd.period_range(start=min_date.to_period('M'), end=max_date.to_period('M'), freq='M')
        results = []
        for month in month_range:
            month_start = month.to_timestamp()
            month_end = (month + 1).to_timestamp()

            # Find active memberships during this month
            # Active = started before or during month AND (no end date OR ended after month start)
            active_mask = (
                (df_memberships_dates['start_date'] <= month_end) &
                ((df_memberships_dates['end_date'].isna()) | (df_memberships_dates['end_date'] >= month_start))
            )
            active_members = df_memberships_dates[active_mask]

            # Split into new (started during this month) vs existing (started before)
            new_mask = (
                (active_members['start_date'] >= month_start) &
                (active_members['start_date'] < month_end)
            )
            new_count = new_mask.sum()
            existing_count = len(active_members) - new_count

            results.append({
                'month': month_start,
                'New That Month': new_count,
                'Existing': existing_count,
                'Total': len(active_members)
            })

        df_composition = pd.DataFrame(results)

        # Create stacked area chart
        fig_composition = go.Figure()

        # Add existing memberships (bottom layer)
        fig_composition.add_trace(
            go.Scatter(
                x=df_composition['month'],
                y=df_composition['Existing'],
                mode='lines',
                name='Existing Members',
                line=dict(width=0.5, color=COLORS['primary']),  # Rust
                stackgroup='one',
                fillcolor=COLORS['primary'],
                hovertemplate='<b>Existing:</b> %{y}<extra></extra>'
            )
        )

        # Add new memberships (top layer)
        fig_composition.add_trace(
            go.Scatter(
                x=df_composition['month'],
                y=df_composition['New That Month'],
                mode='lines',
                name='New That Month',
                line=dict(width=0.5, color=COLORS['secondary']),  # Gold
                stackgroup='one',
                fillcolor=COLORS['secondary'],
                hovertemplate='<b>New:</b> %{y}<extra></extra>'
            )
        )

        fig_composition.update_layout(
            title='Active Memberships Composition: New vs Existing',
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            height=500,
            xaxis_title='Month',
            yaxis_title='Number of Active Memberships',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )

        fig_composition = apply_axis_styling(fig_composition)
        st.plotly_chart(fig_composition, use_container_width=True)

        # Summary metrics
        if not df_composition.empty:
            latest_month = df_composition.iloc[-1]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Current Total Active', int(latest_month['Total']))
            with col2:
                st.metric('New This Month', int(latest_month['New That Month']))
            with col3:
                pct_new = (latest_month['New That Month'] / latest_month['Total'] * 100) if latest_month['Total'] > 0 else 0
                st.metric('% New This Month', f"{pct_new:.1f}%")
    else:
        st.info('No membership data available for composition analysis')

    # At-Risk Members Table
    st.subheader('At-Risk Members')

    if not df_at_risk.empty:
        risk_category_filter = st.multiselect(
            'Filter by Risk Category',
            options=df_at_risk['risk_category'].unique(),
            default=df_at_risk['risk_category'].unique()
        )

        df_at_risk_filtered = df_at_risk_display[
            df_at_risk['risk_category'].isin(risk_category_filter)
        ]

        st.dataframe(
            df_at_risk_filtered,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Capitan Link': st.column_config.LinkColumn('Capitan Link')
            }
        )

        st.caption(f'Total at-risk members: {len(df_at_risk_filtered)}')

    # New Members Table
    st.subheader('New Members (Last 28 Days)')
    st.markdown('Members who joined in the last 28 days, sorted by most recent first')

    if not df_new_members.empty:
        # Prepare display dataframe
        df_new_members_display = df_new_members.copy()

        # Format dates
        if 'start_date' in df_new_members_display.columns:
            df_new_members_display['start_date'] = pd.to_datetime(df_new_members_display['start_date']).dt.strftime('%Y-%m-%d')

        # Select and rename columns for display
        display_columns = {
            'customer_id': 'Customer ID',
            'first_name': 'First Name',
            'last_name': 'Last Name',
            'age': 'Age',
            'membership_type': 'Membership Type',
            'start_date': 'Start Date',
            'days_since_joining': 'Days Since Joining',
            'total_checkins': 'Total Check-ins',
            'capitan_link': 'Capitan Link'
        }

        # Filter to only columns that exist
        available_columns = {k: v for k, v in display_columns.items() if k in df_new_members_display.columns}
        df_new_members_display = df_new_members_display[list(available_columns.keys())]
        df_new_members_display.columns = list(available_columns.values())

        st.dataframe(
            df_new_members_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Capitan Link': st.column_config.LinkColumn('Capitan Link')
            }
        )

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total New Members', len(df_new_members))
        with col2:
            avg_checkins = df_new_members['total_checkins'].mean() if 'total_checkins' in df_new_members.columns else 0
            st.metric('Avg Check-ins per New Member', f"{avg_checkins:.1f}")
        with col3:
            # Count by membership type
            if 'membership_type' in df_new_members.columns:
                top_type = df_new_members['membership_type'].value_counts().index[0]
                top_count = df_new_members['membership_type'].value_counts().iloc[0]
                st.metric('Most Common Type', f"{top_type} ({top_count})")

        st.caption(f'Generated: {df_new_members["generated_at"].iloc[0] if "generated_at" in df_new_members.columns else "N/A"}')
    else:
        st.info('No new members found in the last 28 days')

    # Recently Attrited Members
    st.subheader('Recently Attrited Members (Last 60 Days)')

    # Find members whose membership ended in last 60 days and no longer have active membership
    from datetime import datetime, timedelta
    sixty_days_ago = datetime.now() - timedelta(days=60)

    # Get members whose membership ended recently
    df_memberships['end_date'] = pd.to_datetime(df_memberships['end_date'], errors='coerce')
    recently_ended = df_memberships[
        (df_memberships['end_date'] >= sixty_days_ago) &
        (df_memberships['end_date'] <= datetime.now())
    ].copy()

    # Check if they have any active memberships
    active_member_ids = df_memberships[
        df_memberships['status'] == 'ACT'
    ]['owner_id'].unique()

    # Filter to only those without active memberships
    attrited = recently_ended[
        ~recently_ended['owner_id'].isin(active_member_ids)
    ].copy()

    if not attrited.empty:
        # Get unique members (they might have multiple ended memberships)
        attrited_unique = attrited.sort_values('end_date', ascending=False).drop_duplicates('owner_id')

        # Join with members table to get names
        # Match owner_id from memberships to customer_id in members
        attrited_with_names = attrited_unique.merge(
            df_members[['customer_id', 'member_first_name', 'member_last_name']],
            left_on='owner_id',
            right_on='customer_id',
            how='left'
        )

        # Prepare display data
        attrited_display = attrited_with_names[[
            'owner_id', 'member_first_name', 'member_last_name', 'membership_owner_age', 'name', 'end_date'
        ]].copy()
        attrited_display.columns = ['Customer ID', 'First Name', 'Last Name', 'Age', 'Membership Type', 'End Date']

        # Add Capitan link
        attrited_display['Capitan Link'] = attrited_display['Customer ID'].apply(
            lambda x: f"https://app.hellocapitan.com/customers/{x}/check-ins" if pd.notna(x) else ''
        )

        # Format end date
        attrited_display['End Date'] = pd.to_datetime(attrited_display['End Date']).dt.strftime('%Y-%m-%d')

        st.dataframe(
            attrited_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Capitan Link': st.column_config.LinkColumn('Capitan Link')
            }
        )

        st.caption(f'Total recently attrited members: {len(attrited_unique)}')
    else:
        st.info('No recently attrited members found')

    # ========== COHORT RETENTION ==========
    st.subheader('Membership Cohort Retention')
    st.markdown('For each month cohort (when members joined), how long did they stay? How many are still active?')

    df_cohort = df_memberships.copy()
    df_cohort['start_date'] = pd.to_datetime(df_cohort.get('start_date'), errors='coerce')
    df_cohort['end_date'] = pd.to_datetime(df_cohort.get('end_date'), errors='coerce')
    df_cohort = df_cohort[df_cohort['start_date'].notna()].copy()

    if not df_cohort.empty:
        cohort_today = pd.Timestamp.now()
        df_cohort['cohort_month'] = df_cohort['start_date'].dt.to_period('M')

        # Duration in months
        df_cohort['duration_months'] = df_cohort.apply(
            lambda r: max(0, (cohort_today - r['start_date']).days / 30.44)
            if r['status'] == 'ACT'
            else max(0, (r['end_date'] - r['start_date']).days / 30.44)
            if pd.notna(r['end_date'])
            else 0,
            axis=1
        )

        # Build retention matrix
        cohort_months = sorted(df_cohort['cohort_month'].unique())
        max_months_to_show = 12

        retention_data = []
        for cm in cohort_months:
            cohort = df_cohort[df_cohort['cohort_month'] == cm]
            total = len(cohort)
            still_active = len(cohort[cohort['status'] == 'ACT'])
            row = {
                'Cohort': str(cm),
                'Total': total,
                'Still Active': still_active,
                'Active %': round(still_active / total * 100) if total > 0 else 0
            }
            for m in range(1, max_months_to_show + 1):
                retained = len(cohort[cohort['duration_months'] >= m])
                row[f'M{m}'] = round(retained / total * 100) if total > 0 else 0
            retention_data.append(row)

        df_retention = pd.DataFrame(retention_data)

        # Summary: still active by cohort
        st.markdown('**Members Still Active by Cohort**')
        df_active_summary = df_retention[['Cohort', 'Total', 'Still Active', 'Active %']].copy()
        # Only show last 18 months
        df_active_summary = df_active_summary.tail(18)

        fig_active = go.Figure()
        fig_active.add_trace(go.Bar(
            x=df_active_summary['Cohort'],
            y=df_active_summary['Total'],
            name='Total Joined',
            marker_color=COLORS['secondary'],
            text=df_active_summary['Total'],
            textposition='outside',
            textfont=dict(size=14)
        ))
        fig_active.add_trace(go.Bar(
            x=df_active_summary['Cohort'],
            y=df_active_summary['Still Active'],
            name='Still Active',
            marker_color=COLORS['quaternary'],
            text=df_active_summary['Still Active'],
            textposition='outside',
            textfont=dict(size=14)
        ))
        fig_active.update_layout(
            title='Cohort Size vs Still Active Members',
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            height=450,
            barmode='group',
            xaxis_title='Join Month',
            yaxis_title='Members',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig_active = apply_axis_styling(fig_active)
        st.plotly_chart(fig_active, use_container_width=True)

        # Retention heatmap
        st.markdown('**Retention Rate by Month Since Joining (%)**')
        month_cols = [f'M{m}' for m in range(1, max_months_to_show + 1)]
        heatmap_cols = month_cols + ['Still Member']
        df_heatmap = df_retention[['Cohort', 'Total'] + month_cols + ['Active %']].tail(18).copy()
        df_heatmap = df_heatmap.rename(columns={'Active %': 'Still Member'})
        df_heatmap['Cohort Label'] = df_heatmap['Cohort'] + '  (' + df_heatmap['Total'].astype(str) + ' joined)'
        cohort_labels = df_heatmap['Cohort Label'].tolist()
        df_heatmap = df_heatmap.drop(columns=['Total', 'Cohort Label']).set_index('Cohort')

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=df_heatmap.values,
            x=heatmap_cols,
            y=cohort_labels,
            colorscale=[
                [0, '#FFFFFF'],
                [0.5, '#BAA052'],
                [1, '#1A2E31']
            ],
            text=df_heatmap.values,
            texttemplate='%{text}%',
            textfont=dict(size=15),
            hoverongaps=False,
            colorbar=dict(title='Retained %', titlefont=dict(size=14), tickfont=dict(size=13))
        ))
        fig_heatmap.update_layout(
            title=dict(text='Cohort Retention Heatmap (% still members at month N)', font=dict(size=18)),
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            height=max(500, len(df_heatmap) * 38 + 120),
            xaxis_title='Months Since Joining',
            yaxis_title='Join Month (Cohort Size)',
            xaxis=dict(tickfont=dict(size=14), title_font=dict(size=16)),
            yaxis=dict(autorange='reversed', tickfont=dict(size=14), title_font=dict(size=16))
        )
        fig_heatmap = apply_axis_styling(fig_heatmap)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Expandable detail table
        with st.expander('View Retention Data Table'):
            display_cols = ['Cohort', 'Total', 'Still Active', 'Active %'] + month_cols
            st.dataframe(df_retention[display_cols].tail(18), use_container_width=True, hide_index=True)
    else:
        st.info('No membership data available for cohort analysis')

# ============================================================================
# TAB 3: DAY PASSES & CHECK-INS
# ============================================================================
with tab3:
    st.header('Day Passes & Check-ins')

    # Timeframe selector for day pass charts
    timeframe_daypass = st.selectbox(
        'Select Timeframe',
        options=['D', 'W', 'M', 'Y'],
        format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Y': 'Yearly'}[x],
        index=2,  # Default to Monthly
        key='daypass_timeframe'
    )

    # Day Pass Count
    st.subheader('Total Day Passes Purchased')

    df_day_pass = df_transactions[df_transactions['revenue_category'] == 'Day Pass'].copy()
    df_day_pass['Date'] = pd.to_datetime(df_day_pass['Date'], errors='coerce')
    df_day_pass = df_day_pass[df_day_pass['Date'].notna()]
    df_day_pass['date'] = df_day_pass['Date'].dt.to_period(timeframe_daypass).dt.start_time

    day_pass_sum = (
        df_day_pass.groupby('date')['Day Pass Count']
        .sum()
        .reset_index(name='total_day_passes')
    )

    # Calculate total for caption
    total_day_passes = day_pass_sum['total_day_passes'].sum()

    fig_day_pass_count = px.bar(
        day_pass_sum,
        x='date',
        y='total_day_passes',
        title='Total Day Passes Purchased',
        text=day_pass_sum['total_day_passes']
    )
    fig_day_pass_count.update_traces(
        marker_color=COLORS['quaternary'],
        textposition='outside',
        textfont=dict(size=14)
    )
    fig_day_pass_count.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text'],
        yaxis_title='Number of Day Passes',
        xaxis_title='Date',
        xaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        )
    )
    fig_day_pass_count = apply_axis_styling(fig_day_pass_count)
    st.plotly_chart(fig_day_pass_count, use_container_width=True)
    st.caption(f'Total day passes: {int(total_day_passes):,}')

    # Day Pass Revenue
    st.subheader('Day Pass Revenue')

    day_pass_revenue = (
        df_day_pass.groupby('date')['Total Amount']
        .sum()
        .reset_index(name='revenue')
    )

    fig_day_pass_revenue = px.bar(
        day_pass_revenue,
        x='date',
        y='revenue',
        title='Day Pass Revenue Over Time',
        text=day_pass_revenue['revenue'].apply(lambda x: f'${x/1000:.1f}K')
    )
    fig_day_pass_revenue.update_traces(
        marker_color=COLORS['tertiary'],
        textposition='outside',
        textfont=dict(size=14)
    )
    fig_day_pass_revenue.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text'],
        yaxis_title='Revenue ($)',
        xaxis_title='Date',
        xaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        )
    )
    fig_day_pass_revenue = apply_axis_styling(fig_day_pass_revenue)
    st.plotly_chart(fig_day_pass_revenue, use_container_width=True)

    # Day Pass Customer Recency Analysis
    st.subheader('Day Pass Users by Customer Type')

    if not df_checkins.empty:
        try:
            # Get all check-ins and filter to day pass entries
            df_checkins_clean = df_checkins.copy()
            df_checkins_clean['checkin_datetime'] = pd.to_datetime(df_checkins_clean['checkin_datetime'], errors='coerce', utc=True)
            df_checkins_clean = df_checkins_clean[df_checkins_clean['checkin_datetime'].notna()].copy()
            df_checkins_clean['checkin_datetime'] = df_checkins_clean['checkin_datetime'].dt.tz_localize(None)

            # Filter to day pass entries
            day_pass_keywords = ['day pass', 'punch pass', 'pass']
            df_day_pass_checkins = df_checkins_clean[
                df_checkins_clean['entry_method_description'].str.lower().str.contains('|'.join(day_pass_keywords), na=False)
            ].copy()

            if not df_day_pass_checkins.empty:
                # Sort by datetime for chronological analysis
                df_day_pass_checkins = df_day_pass_checkins.sort_values('checkin_datetime')

                # Prepare membership data to check if customer had active membership at time of check-in
                df_memberships_check = df_memberships.copy()
                if 'start_date' in df_memberships_check.columns and 'end_date' in df_memberships_check.columns:
                    df_memberships_check['start_date'] = pd.to_datetime(df_memberships_check['start_date'], errors='coerce')
                    df_memberships_check['end_date'] = pd.to_datetime(df_memberships_check['end_date'], errors='coerce')

                # For each day pass check-in, check if customer was a non-member at that time
                recency_data = []
                for _, checkin in df_day_pass_checkins.iterrows():
                    customer_id = checkin['customer_id']
                    checkin_date = checkin['checkin_datetime']

                    # Check if customer had an active membership at time of check-in
                    was_member = False
                    if 'start_date' in df_memberships_check.columns and 'end_date' in df_memberships_check.columns:
                        customer_memberships = df_memberships_check[df_memberships_check['owner_id'] == customer_id]
                        for _, membership in customer_memberships.iterrows():
                            start = membership['start_date']
                            end = membership['end_date']
                            if pd.notna(start) and pd.notna(end):
                                if start <= checkin_date <= end:
                                    was_member = True
                                    break

                    # Skip if they were a member at time of check-in
                    if was_member:
                        continue

                    # Determine customer type based on prior check-in history (non-member day passes only)
                    prior_checkins = df_checkins_clean[
                        (df_checkins_clean['customer_id'] == customer_id) &
                        (df_checkins_clean['checkin_datetime'] < checkin_date)
                    ]

                    if len(prior_checkins) == 0:
                        recency_category = 'New Customer'
                    else:
                        # Get most recent prior check-in date
                        last_checkin = prior_checkins['checkin_datetime'].max()
                        days_since = (checkin_date - last_checkin).days

                        if days_since <= 60:  # 0-2 months
                            recency_category = 'Returning (0-2mo)'
                        elif days_since <= 180:  # 2-6 months
                            recency_category = 'Returning (2-6mo)'
                        else:  # 6+ months
                            recency_category = 'Returning (6+mo)'

                    recency_data.append({
                        'date': checkin_date,
                        'recency_category': recency_category,
                        'count': 1
                    })

                if recency_data:
                    df_recency = pd.DataFrame(recency_data)

                    # Use Grouper for more reliable date aggregation, especially for weekly periods
                    if timeframe_daypass == 'W':
                        # For weekly, use Sunday as week start (W-SUN)
                        df_recency = df_recency.set_index('date')
                        recency_summary = (
                            df_recency
                            .groupby([pd.Grouper(freq='W-SUN'), 'recency_category'])['count']
                            .sum()
                            .reset_index()
                        )
                        recency_summary.rename(columns={'date': 'date_period'}, inplace=True)
                    else:
                        # For other timeframes, use period approach
                        df_recency['date_period'] = df_recency['date'].dt.to_period(timeframe_daypass).dt.start_time
                        recency_summary = (
                            df_recency
                            .groupby(['date_period', 'recency_category'])['count']
                            .sum()
                            .reset_index()
                        )

                    # Define category order and colors
                    category_order = [
                        'New Customer',
                        'Returning (0-2mo)',
                        'Returning (2-6mo)',
                        'Returning (6+mo)'
                    ]

                    color_map = {
                        'New Customer': COLORS['primary'],          # Rust
                        'Returning (0-2mo)': COLORS['secondary'],   # Gold
                        'Returning (2-6mo)': COLORS['quaternary'],  # Teal
                        'Returning (6+mo)': COLORS['tertiary']      # Sage
                    }

                    fig_recency = px.bar(
                        recency_summary,
                        x='date_period',
                        y='count',
                        color='recency_category',
                        title='Day Pass Check-ins by Customer Type',
                        barmode='stack',
                        category_orders={'recency_category': category_order},
                        color_discrete_map=color_map
                    )

                    fig_recency.update_layout(
                        plot_bgcolor=COLORS['background'],
                        paper_bgcolor=COLORS['background'],
                        font_color=COLORS['text'],
                        yaxis_title='Number of Day Passes',
                        xaxis_title='Date',
                        legend_title='Customer Type',
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=-0.3,
                            xanchor='center',
                            x=0.5
                        )
                    )

                    fig_recency = apply_axis_styling(fig_recency)
                    st.plotly_chart(fig_recency, use_container_width=True)

                    # Show summary stats for the most recent period only
                    if not recency_summary.empty:
                        latest_period = recency_summary['date_period'].max()
                        latest_data = recency_summary[recency_summary['date_period'] == latest_period]
                        total_by_category_recent = latest_data.set_index('recency_category')['count']
                        total_passes_recent = total_by_category_recent.sum()

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            new_pct = 100 * total_by_category_recent.get('New Customer', 0) / total_passes_recent if total_passes_recent > 0 else 0
                            st.metric('New Customers', f"{new_pct:.1f}%", help=f"Most recent period: {latest_period.strftime('%Y-%m-%d')}")
                        with col2:
                            recent_pct = 100 * total_by_category_recent.get('Returning (0-2mo)', 0) / total_passes_recent if total_passes_recent > 0 else 0
                            st.metric('Returning (0-2 mo)', f"{recent_pct:.1f}%", help=f"Most recent period: {latest_period.strftime('%Y-%m-%d')}")
                        with col3:
                            return_pct = 100 * total_by_category_recent.get('Returning (2-6mo)', 0) / total_passes_recent if total_passes_recent > 0 else 0
                            st.metric('Returning (2-6 mo)', f"{return_pct:.1f}%", help=f"Most recent period: {latest_period.strftime('%Y-%m-%d')}")
                        with col4:
                            long_pct = 100 * total_by_category_recent.get('Returning (6+mo)', 0) / total_passes_recent if total_passes_recent > 0 else 0
                            st.metric('Returning (6+ mo)', f"{long_pct:.1f}%", help=f"Most recent period: {latest_period.strftime('%Y-%m-%d')}")

                    # Detailed engagement table
                    st.markdown('---')
                    st.subheader('Day Pass User Engagement Details')

                    if not df_day_pass_engagement.empty:
                        df_engagement = df_day_pass_engagement.copy()

                        # Convert dates
                        df_engagement['latest_day_pass_date'] = pd.to_datetime(df_engagement['latest_day_pass_date'], errors='coerce')
                        df_engagement['previous_visit_date'] = pd.to_datetime(df_engagement['previous_visit_date'], errors='coerce')

                        # Show filters
                        col1, col2 = st.columns(2)
                        with col1:
                            show_recent_only = st.checkbox('Show last 30 days only', value=True, key='day_pass_recent_filter')
                        with col2:
                            min_visits = st.slider('Min visits in last 6 months', 0, 20, 0, key='day_pass_visits_filter')

                        # Apply filters
                        df_filtered = df_engagement.copy()
                        if show_recent_only:
                            cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
                            df_filtered = df_filtered[df_filtered['latest_day_pass_date'] >= cutoff]

                        df_filtered = df_filtered[df_filtered['visits_last_6mo'] >= min_visits]

                        # Format for display
                        if not df_filtered.empty:
                            df_display = df_filtered.copy()

                            # Combine name
                            df_display['name'] = (df_display['customer_first_name'] + ' ' + df_display['customer_last_name']).str.strip()

                            # Format dates and numbers
                            df_display['latest_day_pass_date'] = df_display['latest_day_pass_date'].dt.strftime('%Y-%m-%d')
                            df_display['previous_visit_date'] = df_display['previous_visit_date'].apply(
                                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else 'First Visit'
                            )
                            df_display['days_since_last_visit'] = df_display['days_since_last_visit'].apply(
                                lambda x: f"{int(x)}" if pd.notna(x) else 'N/A'
                            )

                            # Select and rename columns
                            df_display = df_display[[
                                'customer_id', 'name', 'customer_email', 'latest_day_pass_date',
                                'previous_visit_date', 'days_since_last_visit',
                                'visits_last_2mo', 'visits_last_6mo', 'visits_last_12mo'
                            ]]
                            df_display.columns = [
                                'Customer ID', 'Name', 'Email', 'Latest Day Pass',
                                'Previous Visit', 'Days Since', 'Visits (2mo)', 'Visits (6mo)', 'Visits (12mo)'
                            ]

                            st.dataframe(
                                df_display,
                                use_container_width=True,
                                hide_index=True,
                                height=400
                            )

                            st.caption(f'Showing {len(df_filtered):,} of {len(df_engagement):,} total non-member day pass users')
                        else:
                            st.info('No customers match the current filters')
                    else:
                        st.info('Engagement data not available')

                else:
                    st.info('No day pass check-ins to analyze')
            else:
                st.info('No day pass check-ins found')

        except Exception as e:
            st.error(f'Error analyzing day pass check-ins: {str(e)}')
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info('Check-in data required for day pass analysis')

    # Membership Conversion Funnel
    st.markdown('---')
    st.subheader('Path to Membership: Check-ins Before Joining')
    st.markdown('How many visits did new members have before purchasing their first membership?')

    if not df_membership_conversion.empty:
        df_conversion = df_membership_conversion.copy()
        df_conversion['membership_start_date'] = pd.to_datetime(df_conversion['membership_start_date'], errors='coerce')

        # Filter to only memberships that have already started (no future memberships)
        today = pd.Timestamp.now()
        df_conversion = df_conversion[df_conversion['membership_start_date'] <= today].copy()

        if not df_conversion.empty:
            # Aggregate by time period for trend analysis
            df_conversion['period'] = df_conversion['membership_start_date'].dt.to_period(timeframe_daypass).dt.start_time

            # Calculate average check-ins per period
            avg_by_period = df_conversion.groupby('period')['previous_checkins_count'].mean().reset_index()
            avg_by_period.columns = ['period', 'avg_checkins']

            # Distribution by bucket over time
            bucket_by_period = df_conversion.groupby(['period', 'checkins_bucket']).size().reset_index(name='count')

            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_overall = df_conversion['previous_checkins_count'].mean()
                st.metric('Avg Check-ins Before Membership', f"{avg_overall:.1f}")
            with col2:
                median_overall = df_conversion['previous_checkins_count'].median()
                st.metric('Median Check-ins', f"{int(median_overall)}")
            with col3:
                zero_checkins = len(df_conversion[df_conversion['previous_checkins_count'] == 0])
                zero_pct = 100 * zero_checkins / len(df_conversion)
                st.metric('Joined Without Visiting', f"{zero_pct:.1f}%")
            with col4:
                st.metric('Total New Memberships', f"{len(df_conversion):,}")

            # Line chart: Average check-ins over time
            if not avg_by_period.empty:
                fig_avg = px.line(
                    avg_by_period,
                    x='period',
                    y='avg_checkins',
                    title='Average Check-ins Before Membership (Over Time)',
                    markers=True
                )
                fig_avg.update_traces(line_color=COLORS['primary'], line_width=3)
                fig_avg.update_layout(
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['background'],
                    font_color=COLORS['text'],
                    yaxis_title='Average Check-ins',
                    xaxis_title='Membership Start Date',
                    showlegend=False
                )
                fig_avg = apply_axis_styling(fig_avg)
                st.plotly_chart(fig_avg, use_container_width=True)

            # Stacked bar chart: Distribution by bucket
            if not bucket_by_period.empty:
                # Define bucket order
                bucket_order = ['0', '1', '2', '3', '4', '5+']

                # Define colors for buckets (gradient from new to experienced)
                color_map = {
                    '0': COLORS['primary'],      # Rust - brand new
                    '1': COLORS['secondary'],    # Gold
                    '2': COLORS['quaternary'],   # Teal
                    '3': COLORS['tertiary'],     # Sage
                    '4': '#8B7355',              # Muted brown
                    '5+': '#4A4A4A'              # Dark gray - experienced
                }

                fig_dist = px.bar(
                    bucket_by_period,
                    x='period',
                    y='count',
                    color='checkins_bucket',
                    title='New Members by Previous Check-in Count',
                    barmode='stack',
                    category_orders={'checkins_bucket': bucket_order},
                    color_discrete_map=color_map
                )
                fig_dist.update_layout(
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['background'],
                    font_color=COLORS['text'],
                    yaxis_title='Number of New Members',
                    xaxis_title='Membership Start Date',
                    legend_title='Check-ins Before',
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=-0.3,
                        xanchor='center',
                        x=0.5
                    )
                )
                fig_dist = apply_axis_styling(fig_dist)
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info('No memberships have started yet')

    else:
        st.info('Conversion metrics not available')

    # Day Passes Used (from checkins)
    st.subheader('Day Passes Used (Check-ins)')

    if not df_checkins.empty:
        # Filter checkins to day pass entries only
        df_day_pass_checkins = df_checkins.copy()
        df_day_pass_checkins['checkin_datetime'] = pd.to_datetime(df_day_pass_checkins['checkin_datetime'], errors='coerce', utc=True)
        df_day_pass_checkins = df_day_pass_checkins[df_day_pass_checkins['checkin_datetime'].notna()].copy()
        df_day_pass_checkins['checkin_datetime'] = df_day_pass_checkins['checkin_datetime'].dt.tz_localize(None)

        # Filter for day pass entry methods
        day_pass_keywords = ['day pass', 'punch pass', 'pass']
        df_day_pass_checkins = df_day_pass_checkins[
            df_day_pass_checkins['entry_method_description'].str.lower().str.contains('|'.join(day_pass_keywords), na=False)
        ]

        # Use Grouper for more reliable date aggregation, especially for weekly periods
        if timeframe_daypass == 'W':
            # For weekly, use Sunday as week start (W-SUN)
            df_day_pass_checkins = df_day_pass_checkins.set_index('checkin_datetime')
            day_pass_used_summary = (
                df_day_pass_checkins
                .groupby(pd.Grouper(freq='W-SUN'))
                .size()
                .reset_index(name='passes_used')
            )
            day_pass_used_summary.rename(columns={'checkin_datetime': 'date'}, inplace=True)
        else:
            # For other timeframes, use period approach
            df_day_pass_checkins['date'] = df_day_pass_checkins['checkin_datetime'].dt.to_period(timeframe_daypass).dt.start_time
            day_pass_used_summary = (
                df_day_pass_checkins.groupby('date')
                .size()
                .reset_index(name='passes_used')
            )

        fig_day_pass_used = px.bar(
            day_pass_used_summary,
            x='date',
            y='passes_used',
            title='Day Passes Used (Actual Check-ins)',
            text='passes_used'
        )
        fig_day_pass_used.update_traces(
            marker_color=COLORS['tertiary'],
            textposition='outside',
            textfont=dict(size=14)
        )
        fig_day_pass_used.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            yaxis_title='Number of Passes Used',
            xaxis_title='Date'
        )
        fig_day_pass_used = apply_axis_styling(fig_day_pass_used)
        st.plotly_chart(fig_day_pass_used, use_container_width=True)

        total_used = day_pass_used_summary['passes_used'].sum()
        st.caption(f'Total day passes used: {int(total_used):,}')
    else:
        st.info('No check-in data available')

    # Check-ins by Member vs Non-Member
    st.subheader('Check-ins: Members vs Non-Members')

    if not df_checkins.empty:
        # Prepare check-in data
        df_checkins_chart = df_checkins.copy()
        df_checkins_chart['checkin_datetime'] = pd.to_datetime(df_checkins_chart['checkin_datetime'], errors='coerce', utc=True)
        df_checkins_chart = df_checkins_chart[df_checkins_chart['checkin_datetime'].notna()].copy()
        df_checkins_chart['checkin_datetime'] = df_checkins_chart['checkin_datetime'].dt.tz_localize(None)

        # Determine if check-in is from member or non-member
        # Check if customer has a membership (is in memberships df with active status)
        active_member_customer_ids = set()
        if 'owner_id' in df_memberships.columns:
            active_member_customer_ids = set(df_memberships[df_memberships['status'] == 'ACT']['owner_id'].dropna())

        # If we have customer_id in members df, also use that
        if 'customer_id' in df_members.columns:
            member_customer_ids = set(df_members['customer_id'].dropna())
            active_member_customer_ids = active_member_customer_ids.union(member_customer_ids)
        elif 'member_id' in df_members.columns:
            member_customer_ids = set(df_members['member_id'].dropna())
            active_member_customer_ids = active_member_customer_ids.union(member_customer_ids)

        df_checkins_chart['type'] = df_checkins_chart['customer_id'].apply(
            lambda x: 'Member' if x in active_member_customer_ids else 'Non-Member'
        )

        # Use Grouper for more reliable date aggregation, especially for weekly periods
        if timeframe_daypass == 'W':
            # For weekly, use Sunday as week start (W-SUN)
            df_checkins_chart = df_checkins_chart.set_index('checkin_datetime')
            checkins_by_type = (
                df_checkins_chart
                .groupby([pd.Grouper(freq='W-SUN'), 'type'])
                .size()
                .reset_index(name='count')
            )
            checkins_by_type.rename(columns={'checkin_datetime': 'date'}, inplace=True)
        else:
            # For other timeframes, use period approach
            df_checkins_chart['date'] = df_checkins_chart['checkin_datetime'].dt.to_period(timeframe_daypass).dt.start_time
            checkins_by_type = df_checkins_chart.groupby(['date', 'type']).size().reset_index(name='count')

        fig_checkins_type = px.bar(
            checkins_by_type,
            x='date',
            y='count',
            color='type',
            title='Check-ins by Member vs Non-Member',
            barmode='stack',
            color_discrete_map={'Member': COLORS['primary'], 'Non-Member': COLORS['quaternary']}
        )
        fig_checkins_type.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            yaxis_title='Number of Check-ins',
            xaxis_title='Date',
            legend_title='Type'
        )
        fig_checkins_type = apply_axis_styling(fig_checkins_type)
        st.plotly_chart(fig_checkins_type, use_container_width=True)
    else:
        st.info('No check-in data available')

    # Check-ins by Day of Week and Month
    st.subheader('Check-ins by Day of Week and Month')

    if not df_checkins.empty:
        # Group by month and day of week
        df_checkins_dow = df_checkins.copy()
        df_checkins_dow['checkin_datetime'] = pd.to_datetime(df_checkins_dow['checkin_datetime'], errors='coerce', utc=True)
        df_checkins_dow = df_checkins_dow[df_checkins_dow['checkin_datetime'].notna()].copy()
        df_checkins_dow['checkin_datetime'] = df_checkins_dow['checkin_datetime'].dt.tz_localize(None)
        df_checkins_dow['month'] = df_checkins_dow['checkin_datetime'].dt.to_period('M').astype(str)
        df_checkins_dow['day_of_week'] = df_checkins_dow['checkin_datetime'].dt.day_name()

        # Order days of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Distinct color palette for days of week
        day_colors = {
            'Monday': '#1A2E31',     # Dark teal
            'Tuesday': '#8B4229',    # Rust
            'Wednesday': '#BAA052',  # Gold
            'Thursday': '#96A682',   # Sage
            'Friday': '#C85A3E',     # Light rust
            'Saturday': '#2C6B4F',   # Forest green
            'Sunday': '#5B3A6B',     # Purple
        }

        # Group by day of week and month
        checkins_dow_summary = df_checkins_dow.groupby(['day_of_week', 'month']).size().reset_index(name='count')

        # Summary insights: average check-ins per day of week
        avg_by_day = df_checkins_dow.groupby('day_of_week').size().reindex(day_order).reset_index()
        avg_by_day.columns = ['Day', 'Total']
        num_months = df_checkins_dow['month'].nunique()
        avg_by_day['Avg per Month'] = (avg_by_day['Total'] / num_months).round(1)
        busiest_day = avg_by_day.loc[avg_by_day['Total'].idxmax(), 'Day']
        quietest_day = avg_by_day.loc[avg_by_day['Total'].idxmin(), 'Day']

        # Show insight metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Busiest Day', busiest_day, help=f"{avg_by_day[avg_by_day['Day']==busiest_day]['Avg per Month'].iloc[0]:.0f} avg/month")
        with col2:
            st.metric('Quietest Day', quietest_day, help=f"{avg_by_day[avg_by_day['Day']==quietest_day]['Avg per Month'].iloc[0]:.0f} avg/month")
        with col3:
            weekend_total = avg_by_day[avg_by_day['Day'].isin(['Saturday', 'Sunday'])]['Total'].sum()
            weekday_total = avg_by_day[~avg_by_day['Day'].isin(['Saturday', 'Sunday'])]['Total'].sum()
            weekend_pct = (weekend_total / (weekend_total + weekday_total) * 100) if (weekend_total + weekday_total) > 0 else 0
            st.metric('Weekend Share', f'{weekend_pct:.0f}%', help='Percentage of all check-ins on Sat/Sun')

        # Chart 1: Grouped by Day of Week first (x-axis = Day, color = Month)
        fig_checkins_by_day = px.bar(
            checkins_dow_summary,
            x='day_of_week',
            y='count',
            color='month',
            title='Check-ins by Day of Week (Colored by Month)',
            barmode='group',
            category_orders={'day_of_week': day_order}
        )
        fig_checkins_by_day.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            yaxis_title='Number of Check-ins',
            xaxis_title='Day of Week',
            legend_title='Month'
        )
        fig_checkins_by_day = apply_axis_styling(fig_checkins_by_day)
        st.plotly_chart(fig_checkins_by_day, use_container_width=True)

        # Chart 2: Grouped by Month first (x-axis = Month, color = Day of Week)
        st.subheader('Check-ins by Month and Day of Week')

        # Regroup for month-first view
        checkins_month_summary = df_checkins_dow.groupby(['month', 'day_of_week']).size().reset_index(name='count')

        fig_checkins_by_month = px.bar(
            checkins_month_summary,
            x='month',
            y='count',
            color='day_of_week',
            title='Check-ins by Month (Colored by Day of Week)',
            barmode='group',
            category_orders={'day_of_week': day_order},
            color_discrete_map=day_colors
        )
        fig_checkins_by_month.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            yaxis_title='Number of Check-ins',
            xaxis_title='Month',
            legend_title='Day of Week'
        )
        fig_checkins_by_month = apply_axis_styling(fig_checkins_by_month)
        st.plotly_chart(fig_checkins_by_month, use_container_width=True)

        # Average check-ins by day of week table
        with st.expander('View Average Check-ins by Day'):
            st.dataframe(avg_by_day, use_container_width=True, hide_index=True)
    else:
        st.info('No check-in data available')

# ============================================================================
# TAB 4: RENTALS
# ============================================================================
with tab4:
    st.header('Rentals')

    # Birthday Parties Booked
    st.subheader('Birthday Parties Booked')

    # Debug: Show sample transactions to understand structure
    with st.expander("Debug: Recent Transactions (Jan 2026)"):
        recent_txns = df_transactions[
            df_transactions['Date'] >= '2026-01-01'
        ][['Date', 'Description', 'Data Source', 'revenue_category', 'sub_category', 'Total Amount']].head(30)
        st.dataframe(recent_txns)

        # Show birthday-related transactions
        st.write("**Birthday-related transactions in Jan 2026:**")
        birthday_txns = df_transactions[
            (df_transactions['Date'] >= '2026-01-01') &
            (df_transactions['Description'].str.contains('birthday', case=False, na=False))
        ][['Date', 'Description', 'Data Source', 'revenue_category', 'sub_category', 'Total Amount']]
        st.dataframe(birthday_txns)

        # Show all unique values
        st.write("Unique revenue_category:", df_transactions['revenue_category'].unique().tolist())
        st.write("Unique sub_category:", df_transactions['sub_category'].unique().tolist())
        st.write("Unique Data Source:", df_transactions['Data Source'].unique().tolist())

    # Combine old birthday data (Calendly) with new Shopify birthday purchases
    # Old: sub_category == 'birthday' with Calendly in description (deposit payments)
    df_birthday_old = df_transactions[
        (df_transactions['sub_category'] == 'birthday') &
        (df_transactions['Description'].str.contains('Calendly', case=False, na=False))
    ].copy()

    # New: Shopify purchases with 'Birthday Party' in product name or revenue_category == 'Event Booking' with birthday in description
    df_birthday_shopify = df_transactions[
        (df_transactions['Description'].str.contains('Birthday Party', case=False, na=False)) |
        ((df_transactions['revenue_category'] == 'Event Booking') &
         (df_transactions['Description'].str.contains('birthday', case=False, na=False)))
    ].copy()

    # Combine both sources and remove duplicates
    df_birthday = pd.concat([df_birthday_old, df_birthday_shopify], ignore_index=True).drop_duplicates()

    df_birthday['Date'] = pd.to_datetime(df_birthday['Date'], errors='coerce')
    df_birthday = df_birthday[df_birthday['Date'].notna()]
    df_birthday['date'] = df_birthday['Date'].dt.to_period(timeframe).dt.start_time

    if not df_birthday.empty:
        # Count number of parties booked
        birthday_count = (
            df_birthday.groupby('date')
            .size()
            .reset_index(name='num_parties')
        )

        fig_birthday_count = px.bar(
            birthday_count,
            x='date',
            y='num_parties',
            title='Number of Birthday Parties Booked',
            text='num_parties'
        )
        fig_birthday_count.update_traces(
            marker_color=COLORS['secondary'],
            textposition='outside',
            textfont=dict(size=14)
        )
        fig_birthday_count.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            yaxis_title='Number of Parties',
            xaxis_title='Date'
        )
        fig_birthday_count = apply_axis_styling(fig_birthday_count)
        st.plotly_chart(fig_birthday_count, use_container_width=True)
        st.caption(f'Total parties booked: {birthday_count["num_parties"].sum()}')
    else:
        st.info('No birthday party data available')

    # Birthday Party Revenue
    st.subheader('Birthday Party Revenue')

    birthday_revenue = (
        df_birthday.groupby('date')['Total Amount']
        .sum()
        .reset_index()
    )

    fig_birthday_revenue = px.line(
        birthday_revenue,
        x='date',
        y='Total Amount',
        title='Birthday Party Revenue'
    )
    fig_birthday_revenue.update_traces(line_color=COLORS['quaternary'])
    fig_birthday_revenue.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text'],
        yaxis_title='Revenue ($)',
        xaxis_title='Date',
        xaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        )
    )
    fig_birthday_revenue = apply_axis_styling(fig_birthday_revenue)
    st.plotly_chart(fig_birthday_revenue, use_container_width=True)

    # All Rental Revenue (Event Booking category)
    st.subheader('All Rental Revenue (Event Bookings)')

    df_rentals = df_transactions[df_transactions['revenue_category'] == 'Event Booking'].copy()
    df_rentals['Date'] = pd.to_datetime(df_rentals['Date'], errors='coerce')
    df_rentals = df_rentals[df_rentals['Date'].notna()]
    df_rentals['date'] = df_rentals['Date'].dt.to_period(timeframe).dt.start_time

    # Group by sub_category
    rental_by_type = (
        df_rentals.groupby(['date', 'sub_category'])['Total Amount']
        .sum()
        .reset_index()
    )

    fig_all_rentals = px.bar(
        rental_by_type,
        x='date',
        y='Total Amount',
        color='sub_category',
        title='All Rental Revenue by Type (Birthday Parties, Events, etc.)',
        barmode='stack',
        color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['quaternary'], '#C85A3E', '#5B3A6B']
    )
    fig_all_rentals.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font_color=COLORS['text'],
        yaxis_title='Revenue ($)',
        xaxis_title='Date',
        legend_title='Rental Type',
        xaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['axis_text'], size=14),
            gridcolor=COLORS['gridline'],
            title_font=dict(color=COLORS['text'], size=14)
        )
    )
    fig_all_rentals = apply_axis_styling(fig_all_rentals)
    st.plotly_chart(fig_all_rentals, use_container_width=True)

# ============================================================================
# TAB 5: PROGRAMMING
# ============================================================================
with tab5:
    st.header('Programming')

    # Youth Team Members
    st.subheader('Youth Team Members Over Time')

    # Extract team type from membership name in df_memberships
    youth_memberships = []
    for _, membership in df_memberships.iterrows():
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

            if pd.notna(start_date) and pd.notna(end_date):
                youth_memberships.append({
                    'team_type': team_type,
                    'start_date': start_date,
                    'end_date': end_date
                })

    if youth_memberships:
        df_youth = pd.DataFrame(youth_memberships)

        min_date = df_youth['start_date'].min()
        max_date = pd.Timestamp.now()
        date_range = pd.date_range(start=min_date, end=max_date, freq='M')

        youth_counts = []
        for date in date_range:
            active_youth = df_youth[
                (df_youth['start_date'] <= date) &
                (df_youth['end_date'] >= date)
            ]
            # Count by team type
            counts_by_type = active_youth['team_type'].value_counts().to_dict()
            youth_counts.append({
                'date': date,
                'Recreation': counts_by_type.get('Recreation', 0),
                'Development': counts_by_type.get('Development', 0),
                'Competitive': counts_by_type.get('Competitive', 0)
            })

        youth_df = pd.DataFrame(youth_counts)

        # Reshape for stacked area chart
        youth_melted = youth_df.melt(id_vars='date',
                                      value_vars=['Recreation', 'Development', 'Competitive'],
                                      var_name='Team Type',
                                      value_name='Members')

        fig_youth = px.area(
            youth_melted,
            x='date',
            y='Members',
            color='Team Type',
            title='Active Youth Team Members by Team Type',
            color_discrete_map={
                'Recreation': COLORS['primary'],
                'Development': COLORS['secondary'],
                'Competitive': COLORS['tertiary']
            }
        )
        fig_youth.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            yaxis_title='Number of Team Members',
            xaxis_title='Date',
            hovermode='x unified'
        )
        fig_youth = apply_axis_styling(fig_youth)
        st.plotly_chart(fig_youth, use_container_width=True)
    else:
        st.info('No youth team data available')

    # Timeframe selector for Programming tab
    timeframe_prog = st.selectbox(
        'Select Timeframe',
        options=['D', 'W', 'M'],
        format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[x],
        index=2,  # Default to Monthly
        key='programming_timeframe'
    )

    # Youth Team Revenue
    st.subheader('Youth Team Revenue')

    df_team_revenue = df_transactions[df_transactions['revenue_category'] == 'Team'].copy()
    df_team_revenue['Date'] = pd.to_datetime(df_team_revenue['Date'], errors='coerce')
    df_team_revenue = df_team_revenue[df_team_revenue['Date'].notna()]
    df_team_revenue['date'] = df_team_revenue['Date'].dt.to_period(timeframe_prog).dt.start_time

    if not df_team_revenue.empty:
        team_revenue = (
            df_team_revenue.groupby('date')['Total Amount']
            .sum()
            .reset_index()
        )

        fig_team_revenue = px.bar(
            team_revenue,
            x='date',
            y='Total Amount',
            title='Youth Team Revenue',
            text=team_revenue['Total Amount'].apply(lambda x: f'${x/1000:.1f}K')
        )
        fig_team_revenue.update_traces(
            marker_color=COLORS['primary'],
            textposition='outside',
            textfont=dict(size=14)
        )
        fig_team_revenue.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            yaxis_title='Revenue ($)',
            xaxis_title='Date'
        )
        fig_team_revenue = apply_axis_styling(fig_team_revenue)
        st.plotly_chart(fig_team_revenue, use_container_width=True)
    else:
        st.info('No youth team revenue data available')

    # Fitness Revenue
    st.subheader('Fitness Revenue')

    if 'fitness_amount' in df_transactions.columns:
        df_fitness = df_transactions[df_transactions['fitness_amount'] > 0].copy()
        df_fitness['Date'] = pd.to_datetime(df_fitness['Date'], errors='coerce')
        df_fitness = df_fitness[df_fitness['Date'].notna()]
        df_fitness['date'] = df_fitness['Date'].dt.to_period(timeframe_prog).dt.start_time

        fitness_revenue = (
            df_fitness.groupby('date')['fitness_amount']
            .sum()
            .reset_index()
        )

        fig_fitness = px.bar(
            fitness_revenue,
            x='date',
            y='fitness_amount',
            title='Fitness Revenue (Classes, Fitness-Only Memberships, Add-ons)',
            text=fitness_revenue['fitness_amount'].apply(lambda x: f'${x/1000:.1f}K')
        )
        fig_fitness.update_traces(
            marker_color=COLORS['secondary'],
            textposition='outside',
            textfont=dict(size=14)
        )
        fig_fitness.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            yaxis_title='Fitness Revenue ($)',
            xaxis_title='Date'
        )
        fig_fitness = apply_axis_styling(fig_fitness)
        st.plotly_chart(fig_fitness, use_container_width=True)
    else:
        st.info('Fitness revenue data is being calculated. Please wait for the next data pipeline run.')

    # Fitness Class Attendance
    st.subheader('Fitness Class Attendance')

    fitness_event_keywords = ['HYROX', 'transformation', 'strength', 'fitness', 'yoga', 'workout']

    df_events_filtered = df_events.copy()
    df_events_filtered['event_type_name_lower'] = df_events_filtered['event_type_name'].str.lower()

    fitness_mask = df_events_filtered['event_type_name_lower'].apply(
        lambda x: any(keyword.lower() in str(x) for keyword in fitness_event_keywords) if pd.notna(x) else False
    )
    df_events_filtered = df_events_filtered[fitness_mask]

    if not df_events_filtered.empty:
        # Ensure we have a copy to avoid SettingWithCopyWarning
        df_events_filtered = df_events_filtered.copy()
        df_events_filtered['start_datetime'] = pd.to_datetime(df_events_filtered['start_datetime'], errors='coerce', utc=True)
        df_events_filtered = df_events_filtered[df_events_filtered['start_datetime'].notna()].copy()

        if not df_events_filtered.empty and len(df_events_filtered) > 0:
            # Convert timezone-aware datetime to timezone-naive for period conversion
            df_events_filtered['start_datetime'] = df_events_filtered['start_datetime'].dt.tz_localize(None)
            df_events_filtered['date'] = df_events_filtered['start_datetime'].dt.to_period(timeframe_prog).dt.start_time

            attendance = (
                df_events_filtered.groupby('date')['num_reservations']
                .sum()
                .reset_index()
            )

            fig_attendance = px.bar(
                attendance,
                x='date',
                y='num_reservations',
                title='Fitness Class Attendance (Total Reservations)',
                text=attendance['num_reservations'].apply(lambda x: f'{x/1000:.1f}K' if x >= 1000 else str(int(x)))
            )
            fig_attendance.update_traces(
                marker_color=COLORS['tertiary'],
                textposition='outside',
                textfont=dict(size=14)
            )
            fig_attendance.update_layout(
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['background'],
                font_color=COLORS['text'],
                yaxis_title='Total Attendance',
                xaxis_title='Date'
            )
            fig_attendance = apply_axis_styling(fig_attendance)
            st.plotly_chart(fig_attendance, use_container_width=True)

            # Fitness Check-ins by Class Type
            st.subheader('Fitness Check-ins by Class Type')

            # Group by class type (event_type_name or truncated name)
            df_events_by_type = df_events_filtered.copy()

            # Truncate long names for better display
            def truncate_name(name, max_length=30):
                if pd.isna(name):
                    return 'Unknown'
                name_str = str(name)
                if len(name_str) > max_length:
                    return name_str[:max_length-3] + '...'
                return name_str

            df_events_by_type['class_type'] = df_events_by_type['event_type_name'].apply(truncate_name)

            class_type_attendance = (
                df_events_by_type.groupby(['date', 'class_type'])['num_reservations']
                .sum()
                .reset_index()
            )

            fig_class_types = px.bar(
                class_type_attendance,
                x='date',
                y='num_reservations',
                color='class_type',
                title='Fitness Check-ins by Class Type',
                barmode='stack'
            )
            fig_class_types.update_layout(
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['background'],
                font_color=COLORS['text'],
                yaxis_title='Check-ins',
                xaxis_title='Date',
                legend_title='Class Type'
            )
            fig_class_types = apply_axis_styling(fig_class_types)
            st.plotly_chart(fig_class_types, use_container_width=True)
        else:
            st.info('No fitness class data available with valid dates')
    else:
        st.info('No fitness class data available')

    # Camp Signups
    st.subheader('Camp Signups')

    # Filter for camp events
    df_camps = df_events[
        df_events['event_type_name'].str.contains('camp', case=False, na=False)
    ].copy()

    if not df_camps.empty:
        # Parse dates
        df_camps['start_datetime'] = pd.to_datetime(df_camps['start_datetime'], errors='coerce', utc=True)
        df_camps = df_camps[df_camps['start_datetime'].notna()].copy()

        # Convert to timezone-naive for comparisons
        df_camps['start_datetime'] = df_camps['start_datetime'].dt.tz_localize(None)

        # Separate upcoming and past camps
        now = pd.Timestamp.now()
        df_upcoming = df_camps[df_camps['start_datetime'] >= now].copy()
        df_past = df_camps[df_camps['start_datetime'] < now].copy()

        # Display upcoming camps
        st.markdown('#### 🔜 Upcoming Camps')
        if not df_upcoming.empty:
            df_upcoming_display = df_upcoming.sort_values('start_datetime')[[
                'start_datetime', 'event_type_name', 'num_reservations', 'capacity'
            ]].copy()

            # Calculate fill rate
            df_upcoming_display['fill_rate'] = (
                df_upcoming_display['num_reservations'] / df_upcoming_display['capacity'] * 100
            ).round(1)

            # Format for display
            df_upcoming_display['start_datetime'] = df_upcoming_display['start_datetime'].dt.strftime('%Y-%m-%d')
            df_upcoming_display['Signups'] = df_upcoming_display['num_reservations'].astype(int).astype(str) + ' / ' + df_upcoming_display['capacity'].astype(int).astype(str)
            df_upcoming_display['Fill Rate'] = df_upcoming_display['fill_rate'].astype(str) + '%'

            # Select and rename columns for display
            df_upcoming_display = df_upcoming_display[[
                'start_datetime', 'event_type_name', 'Signups', 'Fill Rate'
            ]].copy()
            df_upcoming_display.columns = ['Date', 'Camp Name', 'Signups', 'Fill Rate']

            st.dataframe(df_upcoming_display, use_container_width=True, hide_index=True)

            # Summary stats for upcoming
            total_upcoming_signups = df_upcoming['num_reservations'].sum()
            total_upcoming_capacity = df_upcoming['capacity'].sum()
            avg_fill_rate = (total_upcoming_signups / total_upcoming_capacity * 100) if total_upcoming_capacity > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Total Signups', f'{int(total_upcoming_signups)}')
            with col2:
                st.metric('Total Capacity', f'{int(total_upcoming_capacity)}')
            with col3:
                st.metric('Avg Fill Rate', f'{avg_fill_rate:.1f}%')
        else:
            st.info('No upcoming camps scheduled')

        st.markdown('---')

        # Display past camps
        st.markdown('#### 📅 Past Camps')
        if not df_past.empty:
            df_past_display = df_past.sort_values('start_datetime', ascending=False).head(15)[[
                'start_datetime', 'event_type_name', 'num_reservations', 'capacity'
            ]].copy()

            # Calculate fill rate
            df_past_display['fill_rate'] = (
                df_past_display['num_reservations'] / df_past_display['capacity'] * 100
            ).round(1)

            # Format for display
            df_past_display['start_datetime'] = df_past_display['start_datetime'].dt.strftime('%Y-%m-%d')
            df_past_display['Attendance'] = df_past_display['num_reservations'].astype(int).astype(str) + ' / ' + df_past_display['capacity'].astype(int).astype(str)
            df_past_display['Fill Rate'] = df_past_display['fill_rate'].astype(str) + '%'

            # Select and rename columns for display
            df_past_display = df_past_display[[
                'start_datetime', 'event_type_name', 'Attendance', 'Fill Rate'
            ]].copy()
            df_past_display.columns = ['Date', 'Camp Name', 'Attendance', 'Fill Rate']

            st.dataframe(df_past_display, use_container_width=True, hide_index=True)

            # Chart: Past camp attendance over time
            df_past_chart = df_past.copy()
            df_past_chart['date'] = df_past_chart['start_datetime'].dt.to_period('M').dt.start_time

            past_attendance = df_past_chart.groupby('date').agg({
                'num_reservations': 'sum',
                'capacity': 'sum'
            }).reset_index()

            fig_past_camps = go.Figure()
            fig_past_camps.add_trace(go.Bar(
                x=past_attendance['date'],
                y=past_attendance['num_reservations'],
                name='Attendance',
                marker_color=COLORS['primary']
            ))
            fig_past_camps.add_trace(go.Scatter(
                x=past_attendance['date'],
                y=past_attendance['capacity'],
                name='Capacity',
                mode='lines+markers',
                line=dict(color=COLORS['quaternary'], width=2),
                marker=dict(size=8)
            ))

            fig_past_camps.update_layout(
                title='Past Camp Attendance vs Capacity',
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['background'],
                font_color=COLORS['text'],
                yaxis_title='Count',
                xaxis_title='Month',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            fig_past_camps = apply_axis_styling(fig_past_camps)
            st.plotly_chart(fig_past_camps, use_container_width=True)
        else:
            st.info('No past camp data available')
    else:
        st.info('No camp events found')

# ============================================================================
# TAB 6: MARKETING
# ============================================================================
with tab6:
    st.header('Marketing Performance')

    # Timeframe selector
    marketing_timeframe = st.radio(
        'Select Timeframe',
        options=['Day', 'Week', 'Month'],
        index=2,
        key='marketing_timeframe',
        horizontal=True
    )

    timeframe_map = {'Day': 'D', 'Week': 'W', 'Month': 'M'}
    marketing_period = timeframe_map[marketing_timeframe]

    # ========== FACEBOOK ADS SECTION ==========
    st.subheader('Facebook/Instagram Ads Performance')

    if not df_facebook_ads.empty:
        df_ads = df_facebook_ads.copy()
        df_ads['date'] = pd.to_datetime(df_ads['date'], errors='coerce')
        df_ads = df_ads[df_ads['date'].notna()]

        # Add missing columns with default values
        for col in ['registrations', 'add_to_carts', 'link_clicks', 'leads', 'purchases']:
            if col not in df_ads.columns:
                df_ads[col] = 0

        # Aggregate by campaign for lifetime performance
        st.markdown('**Ad Campaigns - Lifetime Performance**')

        campaign_lifetime = df_ads.groupby(['campaign_name', 'campaign_id']).agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'purchases': 'sum',
            'add_to_carts': 'sum',
            'link_clicks': 'sum',
            'leads': 'sum',
            'registrations': 'sum',
            'date': ['min', 'max']  # Get start and end dates
        }).reset_index()

        # Flatten column names
        campaign_lifetime.columns = ['campaign_name', 'campaign_id', 'spend', 'impressions', 'clicks',
                                      'purchases', 'add_to_carts', 'link_clicks', 'leads',
                                      'registrations', 'start_date', 'end_date']

        # Infer campaign objective based on which metric has the most activity
        def infer_objective(row):
            """Infer campaign objective from metrics."""
            if row['purchases'] > 0:
                return 'Purchases', row['purchases'], row['spend'] / row['purchases'] if row['purchases'] > 0 else 0
            elif row['add_to_carts'] > 0:
                return 'Add to Cart', row['add_to_carts'], row['spend'] / row['add_to_carts'] if row['add_to_carts'] > 0 else 0
            elif row['registrations'] > 0:
                return 'Registrations', row['registrations'], row['spend'] / row['registrations'] if row['registrations'] > 0 else 0
            elif row['leads'] > 0:
                return 'Leads', row['leads'], row['spend'] / row['leads'] if row['leads'] > 0 else 0
            elif row['link_clicks'] > 0:
                return 'Link Clicks', row['link_clicks'], row['spend'] / row['link_clicks'] if row['link_clicks'] > 0 else 0
            elif row['clicks'] > 0:
                return 'Clicks', row['clicks'], row['spend'] / row['clicks'] if row['clicks'] > 0 else 0
            else:
                return 'Impressions', row['impressions'], row['spend'] / row['impressions'] if row['impressions'] > 0 else 0

        campaign_lifetime[['objective', 'result_count', 'cost_per_result']] = campaign_lifetime.apply(
            lambda row: pd.Series(infer_objective(row)), axis=1
        )

        # Prepare display
        from datetime import datetime, timedelta
        today = pd.Timestamp.now().normalize()

        ads_display = campaign_lifetime[[
            'campaign_name', 'start_date', 'end_date', 'objective', 'spend', 'result_count', 'cost_per_result',
            'impressions', 'clicks'
        ]].copy()

        # Determine if campaign is active (has activity within last 2 days)
        ads_display['is_active'] = (today - pd.to_datetime(ads_display['end_date'])).dt.days <= 2

        # Format columns
        ads_display['start_date'] = pd.to_datetime(ads_display['start_date']).dt.strftime('%Y-%m-%d')
        ads_display['end_date_formatted'] = ads_display.apply(
            lambda row: 'Active' if row['is_active'] else pd.to_datetime(row['end_date']).strftime('%Y-%m-%d'),
            axis=1
        )
        ads_display['spend'] = ads_display['spend'].apply(lambda x: f'${x:.2f}')
        ads_display['cost_per_result'] = ads_display['cost_per_result'].apply(lambda x: f'${x:.2f}')
        ads_display['result_count'] = ads_display['result_count'].astype(int)

        # Select and rename columns
        ads_display = ads_display[[
            'campaign_name', 'start_date', 'end_date_formatted', 'objective', 'spend', 'result_count',
            'cost_per_result', 'impressions', 'clicks'
        ]]
        ads_display.columns = ['Campaign', 'Start Date', 'Status', 'Objective', 'Total Spend', 'Results',
                                'Cost per Result', 'Impressions', 'Clicks']

        # Sort by start date descending (most recent first)
        ads_display = ads_display.sort_values('Start Date', ascending=False)

        st.dataframe(ads_display, use_container_width=True, hide_index=True)

        # Campaign comparison chart
        st.markdown('**Campaign Performance Comparison**')

        # Sort campaigns by spend for chart
        campaign_lifetime_sorted = campaign_lifetime.sort_values('spend', ascending=False)

        fig_ads = go.Figure()

        # Add spend bars
        fig_ads.add_trace(go.Bar(
            x=campaign_lifetime_sorted['campaign_name'],
            y=campaign_lifetime_sorted['spend'],
            name='Total Spend',
            marker_color=COLORS['primary'],
            yaxis='y',
            text=campaign_lifetime_sorted['spend'].apply(lambda x: f'${x:.0f}'),
            textposition='outside'
        ))

        # Add cost per result line
        fig_ads.add_trace(go.Scatter(
            x=campaign_lifetime_sorted['campaign_name'],
            y=campaign_lifetime_sorted['cost_per_result'],
            name='Cost per Result',
            line=dict(color=COLORS['secondary'], width=3),
            yaxis='y2',
            mode='lines+markers'
        ))

        fig_ads.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            title='Campaign Spend vs Cost per Result',
            xaxis_title='Campaign',
            yaxis=dict(title='Total Spend ($)', side='left'),
            yaxis2=dict(title='Cost per Result ($)', side='right', overlaying='y'),
            hovermode='x unified',
            showlegend=True
        )

        fig_ads = apply_axis_styling(fig_ads)
        st.plotly_chart(fig_ads, use_container_width=True)
    else:
        st.info('No Facebook Ads data available')

    # ========== INSTAGRAM POSTS SECTION ==========
    st.subheader('Instagram Posts Performance')

    if not df_instagram.empty:
        df_ig = df_instagram.copy()
        df_ig['timestamp'] = pd.to_datetime(df_ig['timestamp'], errors='coerce')
        df_ig = df_ig[df_ig['timestamp'].notna()]

        # Posts metrics over time
        df_ig['period'] = df_ig['timestamp'].dt.to_period(marketing_period).dt.start_time

        ig_by_period = df_ig.groupby('period').agg({
            'post_id': 'count',
            'likes': ['mean', 'min', 'max'],
            'comments': ['mean', 'min', 'max'],
            'reach': ['mean', 'min', 'max'],
            'saved': ['mean', 'min', 'max'],
            'engagement_rate': ['mean', 'min', 'max']
        }).reset_index()

        # Flatten column names
        ig_by_period.columns = ['period', 'num_posts',
                                'likes_avg', 'likes_min', 'likes_max',
                                'comments_avg', 'comments_min', 'comments_max',
                                'reach_avg', 'reach_min', 'reach_max',
                                'saved_avg', 'saved_min', 'saved_max',
                                'engagement_avg', 'engagement_min', 'engagement_max']

        # Chart: Number of posts and average engagement
        fig_ig = make_subplots(specs=[[{"secondary_y": True}]])

        fig_ig.add_trace(
            go.Bar(x=ig_by_period['period'], y=ig_by_period['num_posts'],
                   name='Number of Posts', marker_color=COLORS['primary']),
            secondary_y=False
        )

        fig_ig.add_trace(
            go.Scatter(x=ig_by_period['period'], y=ig_by_period['likes_avg'],
                      name='Avg Likes', line=dict(color=COLORS['secondary'], width=3)),
            secondary_y=True
        )

        fig_ig.add_trace(
            go.Scatter(x=ig_by_period['period'], y=ig_by_period['comments_avg'],
                      name='Avg Comments', line=dict(color=COLORS['tertiary'], width=2)),
            secondary_y=True
        )

        fig_ig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            title=f'Instagram Posts & Engagement by {marketing_timeframe}',
            hovermode='x unified'
        )
        fig_ig.update_yaxes(title_text='Number of Posts', secondary_y=False)
        fig_ig.update_yaxes(title_text='Engagement', secondary_y=True)

        fig_ig = apply_axis_styling(fig_ig)
        st.plotly_chart(fig_ig, use_container_width=True)

        # Stats summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total Posts', len(df_ig))
        with col2:
            st.metric('Avg Likes/Post', f"{df_ig['likes'].mean():.0f}")
        with col3:
            st.metric('Avg Comments/Post', f"{df_ig['comments'].mean():.1f}")
        with col4:
            st.metric('Avg Engagement Rate', f"{df_ig['engagement_rate'].mean():.2f}%")

    else:
        st.info('No Instagram data available')

    # ========== SMS/TWILIO WAIVER REQUESTS SECTION ==========
    st.subheader('SMS Waiver Requests')

    if not df_twilio_messages.empty:
        df_sms = df_twilio_messages.copy()
        df_sms['date_sent'] = pd.to_datetime(df_sms['date_sent'], errors='coerce')
        df_sms = df_sms[df_sms['date_sent'].notna()]

        # Filter to waiver requests only
        df_waiver = df_sms[df_sms['is_waiver_request'] == True].copy()

        if not df_waiver.empty:
            # Aggregate by time period
            df_waiver['period'] = df_waiver['date_sent'].dt.to_period(marketing_period).dt.start_time

            waiver_by_period = df_waiver.groupby('period').agg({
                'message_sid': 'count',
                'from_number': 'nunique'
            }).reset_index()
            waiver_by_period.columns = ['period', 'num_requests', 'unique_numbers']

            # Create bar chart
            fig_sms = go.Figure()

            fig_sms.add_trace(
                go.Bar(
                    x=waiver_by_period['period'],
                    y=waiver_by_period['unique_numbers'],
                    name='Unique Numbers',
                    marker_color=COLORS['primary'],
                    text=waiver_by_period['unique_numbers'],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Unique Numbers: %{y}<br>Total Requests: %{customdata}<extra></extra>',
                    customdata=waiver_by_period['num_requests']
                )
            )

            fig_sms.update_layout(
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['background'],
                font_color=COLORS['text'],
                title=f'Unique Phone Numbers Requesting Waivers by {marketing_timeframe}',
                xaxis_title=marketing_timeframe,
                yaxis_title='Unique Phone Numbers',
                hovermode='x unified',
                height=400
            )

            fig_sms = apply_axis_styling(fig_sms)
            st.plotly_chart(fig_sms, use_container_width=True)

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Total Waiver Requests', len(df_waiver))
            with col2:
                st.metric('Unique Phone Numbers', df_waiver['from_number'].nunique())
            with col3:
                # Calculate most recent request
                most_recent = df_waiver['date_sent'].max()
                # Remove timezone if present
                if pd.notna(most_recent):
                    if hasattr(most_recent, 'tz_localize'):
                        most_recent = most_recent.tz_localize(None)
                    elif hasattr(most_recent, 'tz'):
                        most_recent = most_recent.replace(tzinfo=None)
                    days_ago = (pd.Timestamp.now() - most_recent).days
                    st.metric('Most Recent Request', f'{days_ago} days ago')
                else:
                    st.metric('Most Recent Request', 'N/A')
        else:
            st.info('No waiver request messages found')
    else:
        st.info('No SMS data available')

    # ========== EMAIL CAMPAIGNS SECTION ==========
    st.subheader('Email Campaign Performance')

    if not df_mailchimp.empty:
        df_email = df_mailchimp.copy()
        df_email['send_time'] = pd.to_datetime(df_email['send_time'], errors='coerce')
        df_email = df_email[df_email['send_time'].notna()]

        # Recent campaigns table
        st.markdown('**Recent Email Campaigns**')

        recent_emails = df_email.sort_values('send_time', ascending=False).head(10)

        emails_display = recent_emails[[
            'campaign_title', 'send_time', 'emails_sent', 'open_rate',
            'click_rate', 'unsubscribed'
        ]].copy()

        emails_display['send_time'] = pd.to_datetime(emails_display['send_time']).dt.strftime('%Y-%m-%d')
        emails_display['open_rate'] = emails_display['open_rate'].apply(lambda x: f'{x:.1f}%' if pd.notna(x) else '0%')
        emails_display['click_rate'] = emails_display['click_rate'].apply(lambda x: f'{x:.2f}%' if pd.notna(x) else '0%')
        emails_display.columns = ['Campaign', 'Send Date', 'Emails Sent', 'Open Rate', 'Click Rate', 'Unsubscribed']

        st.dataframe(emails_display, use_container_width=True, hide_index=True)

        # Email performance over time
        df_email['period'] = df_email['send_time'].dt.to_period(marketing_period).dt.start_time

        email_by_period = df_email.groupby('period').agg({
            'campaign_id': 'count',
            'emails_sent': 'sum',
            'open_rate': 'mean',
            'click_rate': 'mean',
            'unsubscribed': 'sum'
        }).reset_index()
        email_by_period.columns = ['period', 'num_campaigns', 'emails_sent', 'avg_open_rate', 'avg_click_rate', 'unsubscribed']

        # Chart: Campaigns and engagement rates
        fig_email = make_subplots(specs=[[{"secondary_y": True}]])

        fig_email.add_trace(
            go.Bar(x=email_by_period['period'], y=email_by_period['num_campaigns'],
                   name='Number of Campaigns', marker_color=COLORS['primary']),
            secondary_y=False
        )

        fig_email.add_trace(
            go.Scatter(x=email_by_period['period'], y=email_by_period['avg_open_rate'],
                      name='Avg Open Rate (%)', line=dict(color=COLORS['secondary'], width=3)),
            secondary_y=True
        )

        fig_email.add_trace(
            go.Scatter(x=email_by_period['period'], y=email_by_period['avg_click_rate'],
                      name='Avg Click Rate (%)', line=dict(color=COLORS['tertiary'], width=2)),
            secondary_y=True
        )

        fig_email.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            title=f'Email Campaigns & Engagement by {marketing_timeframe}',
            hovermode='x unified'
        )
        fig_email.update_yaxes(title_text='Number of Campaigns', secondary_y=False)
        fig_email.update_yaxes(title_text='Rate (%)', secondary_y=True)

        fig_email = apply_axis_styling(fig_email)
        st.plotly_chart(fig_email, use_container_width=True)

        # Stats summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total Campaigns', len(df_email))
        with col2:
            st.metric('Avg Open Rate', f"{df_email['open_rate'].mean():.1f}%")
        with col3:
            st.metric('Avg Click Rate', f"{df_email['click_rate'].mean():.2f}%")
        with col4:
            st.metric('Total Unsubscribes', int(df_email['unsubscribed'].sum()))

    else:
        st.info('No email campaign data available')

    # ========== CUSTOMER FLAGS SECTION ==========
    st.subheader('Customer Engagement Flags')
    st.markdown('Customers flagged for targeted offers and engagement campaigns')

    if not df_customer_flags.empty:
        # Prepare flags data
        df_flags_display = df_customer_flags.copy()
        df_flags_display['triggered_date'] = pd.to_datetime(df_flags_display['triggered_date'], errors='coerce')
        df_flags_display['flag_added_date'] = pd.to_datetime(df_flags_display['flag_added_date'], errors='coerce')

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total Flagged Customers', len(df_flags_display))
        with col2:
            st.metric('Unique Customers', df_flags_display['customer_id'].nunique())
        with col3:
            st.metric('Flag Types', df_flags_display['flag_type'].nunique())
        with col4:
            latest_flag = df_flags_display['flag_added_date'].max()
            st.metric('Latest Flag', latest_flag.strftime('%Y-%m-%d') if pd.notna(latest_flag) else 'N/A')

        # Flag type breakdown
        st.markdown('**Flags by Type**')
        flag_counts = df_flags_display['flag_type'].value_counts().reset_index()
        flag_counts.columns = ['Flag Type', 'Count']

        # Format flag names for display
        flag_counts['Flag Type'] = flag_counts['Flag Type'].str.replace('_', ' ').str.title()

        # Display as table
        st.dataframe(
            flag_counts,
            use_container_width=True,
            hide_index=True
        )

        # Recent flags detail
        st.markdown('**Recent Flags (Last 30 Days)**')
        recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
        df_recent = df_flags_display[df_flags_display['flag_added_date'] >= recent_cutoff].copy()

        if not df_recent.empty:
            # Sort by most recent
            df_recent = df_recent.sort_values('flag_added_date', ascending=False)

            # Select and format columns for display
            display_cols = df_recent[['customer_id', 'flag_type', 'triggered_date', 'priority', 'flag_added_date']].copy()
            display_cols['flag_type'] = display_cols['flag_type'].str.replace('_', ' ').str.title()
            display_cols.columns = ['Customer ID', 'Flag Type', 'Triggered Date', 'Priority', 'Added Date']

            # Format dates
            display_cols['Triggered Date'] = display_cols['Triggered Date'].dt.strftime('%Y-%m-%d')
            display_cols['Added Date'] = display_cols['Added Date'].dt.strftime('%Y-%m-%d %H:%M')

            st.dataframe(
                display_cols,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info('No flags added in the last 30 days')

    else:
        st.info('No customer flags data available')

    # ========== KLAVIYO FLAGS SECTION ==========
    st.subheader('Klaviyo Flow Triggers')
    st.markdown('Flags that trigger automated Klaviyo flows (email/SMS sequences)')

    # Flag types that map to Klaviyo lists/flows
    klaviyo_flag_types = [
        'membership_cancelled_winback',
    ]

    if not df_customer_flags.empty:
        df_klaviyo = df_customer_flags[df_customer_flags['flag_type'].isin(klaviyo_flag_types)].copy()

        if not df_klaviyo.empty:
            df_klaviyo['triggered_date'] = pd.to_datetime(df_klaviyo['triggered_date'], errors='coerce')
            df_klaviyo['flag_added_date'] = pd.to_datetime(df_klaviyo['flag_added_date'], errors='coerce')

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Total Klaviyo Flags', len(df_klaviyo))
            with col2:
                st.metric('Unique Customers', df_klaviyo['customer_id'].nunique())
            with col3:
                latest = df_klaviyo['flag_added_date'].max()
                st.metric('Latest Flag', latest.strftime('%Y-%m-%d') if pd.notna(latest) else 'N/A')

            # Totals by type
            st.markdown('**Total by Flag Type**')
            klaviyo_counts = df_klaviyo['flag_type'].value_counts().reset_index()
            klaviyo_counts.columns = ['Flag Type', 'Count']
            klaviyo_counts['Flag Type'] = klaviyo_counts['Flag Type'].str.replace('_', ' ').str.title()
            st.dataframe(klaviyo_counts, use_container_width=True, hide_index=True)

            # Timeline by type
            st.markdown('**Flags Over Time (by Type)**')
            date_col = 'flag_added_date' if df_klaviyo['flag_added_date'].notna().any() else 'triggered_date'
            df_timeline = df_klaviyo.dropna(subset=[date_col]).copy()

            if not df_timeline.empty:
                df_timeline['date'] = df_timeline[date_col].dt.date
                df_timeline['Flag Type'] = df_timeline['flag_type'].str.replace('_', ' ').str.title()
                daily_counts = df_timeline.groupby(['date', 'Flag Type']).size().reset_index(name='Count')
                daily_counts['date'] = pd.to_datetime(daily_counts['date'])

                import altair as alt
                chart = alt.Chart(daily_counts).mark_bar().encode(
                    x=alt.X('date:T', title='Date'),
                    y=alt.Y('Count:Q', title='Flags Added'),
                    color=alt.Color('Flag Type:N', title='Flag Type'),
                    tooltip=['date:T', 'Flag Type:N', 'Count:Q']
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

            # Detail table
            with st.expander('View Klaviyo Flag Details'):
                detail = df_klaviyo[['customer_id', 'flag_type', 'triggered_date', 'priority', 'flag_added_date']].copy()
                detail['flag_type'] = detail['flag_type'].str.replace('_', ' ').str.title()
                detail.columns = ['Customer ID', 'Flag Type', 'Triggered Date', 'Priority', 'Added Date']
                detail['Triggered Date'] = detail['Triggered Date'].dt.strftime('%Y-%m-%d')
                detail['Added Date'] = detail['Added Date'].dt.strftime('%Y-%m-%d %H:%M')
                detail = detail.sort_values('Added Date', ascending=False)
                st.dataframe(detail, use_container_width=True, hide_index=True)
        else:
            st.info('No Klaviyo flow trigger flags found yet. The membership win-back flag will appear here when members cancel.')
    else:
        st.info('No customer flags data available')

# ============================================================================
# TAB 7: LEAD FLOW (Day Pass → Membership Conversion Funnel)
# ============================================================================
with tab7:
    st.header('🎯 Lead Flow: Day Pass → Membership Funnel (2026+)')
    st.markdown('Track how day pass customers move through the conversion funnel to membership')

    # Filter flags to 2026+ only
    df_flags_2026 = df_customer_flags.copy()
    if not df_flags_2026.empty and 'flag_added_date' in df_flags_2026.columns:
        df_flags_2026['flag_added_date'] = pd.to_datetime(df_flags_2026['flag_added_date'], errors='coerce')
        df_flags_2026 = df_flags_2026[df_flags_2026['flag_added_date'] >= '2026-01-01']

    # ========== ENTRY POINT: DAY PASS CUSTOMERS ==========
    st.subheader('1️⃣ Entry Point: Day Pass Customers')

    # Get total day pass customers from 2026+
    if not df_day_pass_engagement.empty:
        if 'latest_day_pass_date' in df_day_pass_engagement.columns:
            df_day_pass_engagement['latest_day_pass_date'] = pd.to_datetime(df_day_pass_engagement['latest_day_pass_date'], errors='coerce')
            df_day_pass_2026 = df_day_pass_engagement[df_day_pass_engagement['latest_day_pass_date'] >= '2026-01-01']
            total_day_pass_customers = len(df_day_pass_2026)

            # Calculate engagement metrics
            # Multiple visits (came back at least once)
            returning_customers = 0
            if 'total_day_pass_checkins' in df_day_pass_2026.columns:
                returning_customers = len(df_day_pass_2026[df_day_pass_2026['total_day_pass_checkins'] > 1])

            # 2-week membership purchases (look in transactions)
            two_week_purchases = 0
            if not df_transactions.empty and 'customer_id' in df_day_pass_2026.columns and 'customer_id' in df_transactions.columns:
                day_pass_customer_ids = set(df_day_pass_2026['customer_id'].dropna())
                two_week_txns = df_transactions[
                    (df_transactions['customer_id'].isin(day_pass_customer_ids)) &
                    (df_transactions['Description'].str.contains('2-Week|Two Week|2 Week', case=False, na=False))
                ]
                two_week_purchases = two_week_txns['customer_id'].nunique() if not two_week_txns.empty else 0

            # Full membership conversions (look in memberships)
            membership_conversions = 0
            if not df_memberships.empty and 'customer_id' in df_day_pass_2026.columns and 'customer_id' in df_memberships.columns:
                day_pass_customer_ids = set(df_day_pass_2026['customer_id'].dropna())
                active_members = df_memberships[
                    (df_memberships['customer_id'].isin(day_pass_customer_ids)) &
                    (df_memberships['status'] == 'active')
                ]
                membership_conversions = len(active_members)

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Day Pass Customers", f"{total_day_pass_customers:,}")
            with col2:
                st.metric("Returned for 2nd Visit", f"{returning_customers:,}")
            with col3:
                st.metric("Bought 2-Week Pass", f"{two_week_purchases:,}")
            with col4:
                st.metric("Converted to Membership", f"{membership_conversions:,}")

        else:
            st.info("Date information not available")
    else:
        st.info("No day pass engagement data available")

    st.markdown('---')

    # ========== FLAG EVALUATION ==========
    st.subheader('2️⃣ A/B Test Entry: Who Qualifies?')
    st.markdown('Customers who enter the A/B test based on business rules')

    if not df_flags_2026.empty:
        # Filter to only the two A/B test flags
        ab_test_flags = df_flags_2026[df_flags_2026['flag_type'].isin(['first_time_day_pass_2wk_offer', 'second_visit_offer_eligible'])]

        if not ab_test_flags.empty:
            # Count by flag type
            flag_counts = ab_test_flags['flag_type'].value_counts().to_dict()

            col1, col2, col3 = st.columns(3)
            with col1:
                first_time_count = flag_counts.get('first_time_day_pass_2wk_offer', 0)
                st.metric("First-Time 2wk Offer (Group A)", f"{first_time_count:,}")
            with col2:
                second_visit_eligible = flag_counts.get('second_visit_offer_eligible', 0)
                st.metric("2nd Visit Eligible (Group B)", f"{second_visit_eligible:,}")
            with col3:
                total_ab = first_time_count + second_visit_eligible
                st.metric("Total in A/B Test", f"{total_ab:,}")
        else:
            st.info("No A/B test flags found for 2026")
    else:
        st.info("No customer flags data available for 2026")

    # Add flag history expander within section 2
    if not df_customer_events.empty:
        # Filter to flag_set events only
        flag_events = df_customer_events[df_customer_events['event_type'].str.contains('flag_set', na=False)].copy()
        if not flag_events.empty:
            with st.expander(f"📜 Flag History ({len(flag_events):,} flag events)"):
                flag_events_display = flag_events.copy()
                if 'event_date' in flag_events_display.columns:
                    flag_events_display['event_date'] = pd.to_datetime(flag_events_display['event_date'], errors='coerce')
                    flag_events_display = flag_events_display.sort_values('event_date', ascending=False)

                # Select columns for display
                display_cols = ['customer_id', 'event_type', 'event_date', 'event_source']
                available_cols = [c for c in display_cols if c in flag_events_display.columns]
                flag_events_display = flag_events_display[available_cols].head(100)

                flag_events_display.columns = [c.replace('_', ' ').title() for c in flag_events_display.columns]
                st.dataframe(flag_events_display, use_container_width=True, hide_index=True)
                st.caption("Showing most recent 100 flag events")

    st.markdown('---')

    # ========== SHOPIFY SYNC ==========
    st.subheader('3️⃣ Shopify Sync Status')

    if not df_shopify_synced_flags.empty:
        # Show sync statistics
        total_synced = len(df_shopify_synced_flags)
        unique_customers = df_shopify_synced_flags['capitan_customer_id'].nunique() if 'capitan_customer_id' in df_shopify_synced_flags.columns else 0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Synced Tags", f"{total_synced:,}")
        with col2:
            st.metric("Unique Customers Synced", f"{unique_customers:,}")

        # Show breakdown by tag
        if 'tag_name' in df_shopify_synced_flags.columns:
            tag_counts = df_shopify_synced_flags['tag_name'].value_counts()
            st.markdown("**Tags Synced to Shopify:**")
            for tag, count in tag_counts.items():
                st.write(f"  - `{tag}`: {count:,} customers")

        # Expandable detail view
        with st.expander("View Sync Details"):
            df_sync_display = df_shopify_synced_flags.copy()
            if 'synced_at' in df_sync_display.columns:
                df_sync_display = df_sync_display.sort_values('synced_at', ascending=False)
            st.dataframe(df_sync_display.head(50), use_container_width=True, hide_index=True)
            st.caption("Showing most recent 50 syncs")
    else:
        st.info("No Shopify sync tracking data available yet. Run the flag sync pipeline to populate this data.")

    st.markdown('---')

    # ========== DISTRIBUTION CHANNELS ==========
    st.subheader('4️⃣ Distribution Channels')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('**📧 Group A → SendGrid Emails**')
        # This would need email send logs - placeholder for now
        st.info("SendGrid send data not yet integrated")
        # TODO: Add email send log data when available

    with col2:
        st.markdown('**📨 Group B → Mailchimp Tags**')
        st.info("📝 Mailchimp tag data needs to be added to data pipeline. Current data is campaign-level only (opens/clicks), not member-level with tags.")

    st.markdown('---')

    # ========== OFFER RESPONSE: SECOND VISIT AFTER EMAIL ==========
    st.subheader('5️⃣ Offer Response: Second Visit After Email')
    st.markdown('Track which customers who received an offer email came back for a second visit')

    if not df_flags_2026.empty and not df_checkins.empty:
        # Find all customers with "_sent" flags (offers that were actually sent)
        sent_flags = df_flags_2026[df_flags_2026['flag_type'].str.endswith('_sent', na=False)].copy()

        if not sent_flags.empty and 'flag_added_date' in sent_flags.columns and 'customer_id' in sent_flags.columns:
            # Ensure dates are datetime
            sent_flags['flag_added_date'] = pd.to_datetime(sent_flags['flag_added_date'], errors='coerce')
            sent_flags = sent_flags[sent_flags['flag_added_date'].notna()]

            # Prepare checkins data
            df_checkins_analysis = df_checkins.copy()
            if 'checkin_datetime' in df_checkins_analysis.columns:
                df_checkins_analysis['checkin_datetime'] = pd.to_datetime(df_checkins_analysis['checkin_datetime'], errors='coerce')
                df_checkins_analysis = df_checkins_analysis[df_checkins_analysis['checkin_datetime'].notna()]

                # For each customer with a sent flag, check if they had a check-in AFTER the offer was sent
                customers_with_offers = sent_flags['customer_id'].unique()
                returned_after_offer = []

                for customer_id in customers_with_offers:
                    # Get the date the offer was sent (most recent _sent flag for this customer)
                    customer_flags = sent_flags[sent_flags['customer_id'] == customer_id]
                    offer_sent_date = customer_flags['flag_added_date'].max()

                    # Check if they have any check-ins AFTER the offer was sent
                    customer_checkins = df_checkins_analysis[
                        (df_checkins_analysis['customer_id'] == customer_id) &
                        (df_checkins_analysis['checkin_datetime'] > offer_sent_date)
                    ]

                    if not customer_checkins.empty:
                        returned_after_offer.append({
                            'customer_id': customer_id,
                            'offer_sent_date': offer_sent_date,
                            'first_return_date': customer_checkins['checkin_datetime'].min(),
                            'days_to_return': (customer_checkins['checkin_datetime'].min() - offer_sent_date).days,
                            'total_visits_after_offer': len(customer_checkins),
                            'flag_type': customer_flags['flag_type'].iloc[-1]  # Most recent flag
                        })

                df_returned = pd.DataFrame(returned_after_offer)

                # Calculate metrics
                total_offers_sent = len(customers_with_offers)
                total_returned = len(df_returned)
                conversion_rate = (total_returned / total_offers_sent * 100) if total_offers_sent > 0 else 0

                # Breakdown by flag type
                flag_type_breakdown = {}
                if not df_returned.empty:
                    for flag_type in sent_flags['flag_type'].unique():
                        customers_with_this_flag = sent_flags[sent_flags['flag_type'] == flag_type]['customer_id'].unique()
                        returned_with_this_flag = len(df_returned[df_returned['flag_type'] == flag_type])
                        total_with_this_flag = len(customers_with_this_flag)
                        flag_conversion = (returned_with_this_flag / total_with_this_flag * 100) if total_with_this_flag > 0 else 0

                        flag_type_breakdown[flag_type] = {
                            'sent': total_with_this_flag,
                            'returned': returned_with_this_flag,
                            'conversion': flag_conversion
                        }

                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Offers Sent", f"{total_offers_sent:,}")
                with col2:
                    st.metric("Returned for Visit", f"{total_returned:,}")
                with col3:
                    st.metric("Conversion Rate", f"{conversion_rate:.1f}%")

                # Show breakdown by offer type
                if flag_type_breakdown:
                    st.markdown("**Conversion by Offer Type:**")

                    breakdown_data = []
                    for flag_type, stats in flag_type_breakdown.items():
                        # Clean up flag name for display
                        display_name = flag_type.replace('_sent', '').replace('_', ' ').title()
                        breakdown_data.append({
                            'Offer Type': display_name,
                            'Sent': stats['sent'],
                            'Returned': stats['returned'],
                            'Conversion': f"{stats['conversion']:.1f}%"
                        })

                    df_breakdown_display = pd.DataFrame(breakdown_data)
                    st.dataframe(df_breakdown_display, use_container_width=True, hide_index=True)

                # Show details of customers who returned
                if not df_returned.empty:
                    with st.expander(f"View {len(df_returned)} Customers Who Returned"):
                        df_returned_display = df_returned.copy()
                        df_returned_display['offer_sent_date'] = df_returned_display['offer_sent_date'].dt.strftime('%Y-%m-%d')
                        df_returned_display['first_return_date'] = df_returned_display['first_return_date'].dt.strftime('%Y-%m-%d')
                        df_returned_display['flag_type'] = df_returned_display['flag_type'].str.replace('_sent', '').str.replace('_', ' ').str.title()

                        df_returned_display = df_returned_display[[
                            'customer_id', 'flag_type', 'offer_sent_date', 'first_return_date',
                            'days_to_return', 'total_visits_after_offer'
                        ]]
                        df_returned_display.columns = [
                            'Customer ID', 'Offer Type', 'Offer Sent', 'First Return',
                            'Days to Return', 'Total Visits After'
                        ]

                        st.dataframe(df_returned_display, use_container_width=True, hide_index=True)
            else:
                st.info("Check-in datetime information not available")
        else:
            st.info("No offer email sends found for 2026 (looking for flags ending in '_sent')")
    else:
        st.info("Customer flags or check-in data not available")

    st.markdown('---')

    # ========== AB TEST RESULTS ==========
    st.subheader('6️⃣ A/B Test Results: Group A vs Group B')

    if not df_experiment_entries.empty:
        # Show experiment entry statistics
        st.markdown("**Experiment Group Distribution:**")

        if 'experiment_group' in df_experiment_entries.columns:
            group_counts = df_experiment_entries['experiment_group'].value_counts()

            col1, col2, col3 = st.columns(3)
            with col1:
                group_a_count = group_counts.get('A', 0)
                st.metric("Group A (Immediate 2wk Offer)", f"{group_a_count:,}")
            with col2:
                group_b_count = group_counts.get('B', 0)
                st.metric("Group B (Wait for 2nd Visit)", f"{group_b_count:,}")
            with col3:
                total_in_experiment = len(df_experiment_entries)
                st.metric("Total in Experiment", f"{total_in_experiment:,}")

            # Calculate conversion rates if we have checkin data
            if not df_checkins.empty and 'customer_id' in df_experiment_entries.columns:
                st.markdown("**Conversion Analysis (Return Visit Rate):**")

                # Get customer IDs in each group
                group_a_customers = set(df_experiment_entries[df_experiment_entries['experiment_group'] == 'A']['customer_id'].dropna())
                group_b_customers = set(df_experiment_entries[df_experiment_entries['experiment_group'] == 'B']['customer_id'].dropna())

                # Prepare checkins for analysis
                df_checkins_analysis = df_checkins.copy()
                if 'checkin_datetime' in df_checkins_analysis.columns:
                    df_checkins_analysis['checkin_datetime'] = pd.to_datetime(df_checkins_analysis['checkin_datetime'], errors='coerce')

                    # Count customers in each group who had a return visit
                    # (any checkin after their experiment entry date)
                    group_a_returned = 0
                    group_b_returned = 0

                    if 'entry_date' in df_experiment_entries.columns:
                        df_experiment_entries['entry_date'] = pd.to_datetime(df_experiment_entries['entry_date'], errors='coerce')

                        for _, row in df_experiment_entries.iterrows():
                            customer_id = row['customer_id']
                            entry_date = row['entry_date']
                            group = row['experiment_group']

                            if pd.isna(entry_date):
                                continue

                            # Check for checkins after entry date
                            return_visits = df_checkins_analysis[
                                (df_checkins_analysis['customer_id'] == customer_id) &
                                (df_checkins_analysis['checkin_datetime'] > entry_date)
                            ]

                            if not return_visits.empty:
                                if group == 'A':
                                    group_a_returned += 1
                                elif group == 'B':
                                    group_b_returned += 1

                        # Calculate conversion rates
                        group_a_rate = (group_a_returned / group_a_count * 100) if group_a_count > 0 else 0
                        group_b_rate = (group_b_returned / group_b_count * 100) if group_b_count > 0 else 0

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Group A Return Rate",
                                f"{group_a_rate:.1f}%",
                                delta=f"{group_a_returned}/{group_a_count} customers"
                            )
                        with col2:
                            st.metric(
                                "Group B Return Rate",
                                f"{group_b_rate:.1f}%",
                                delta=f"{group_b_returned}/{group_b_count} customers"
                            )

                        # Show winner
                        if group_a_rate > group_b_rate:
                            st.success(f"🏆 Group A (Immediate Offer) is performing {group_a_rate - group_b_rate:.1f}% better")
                        elif group_b_rate > group_a_rate:
                            st.success(f"🏆 Group B (Wait Strategy) is performing {group_b_rate - group_a_rate:.1f}% better")
                        else:
                            st.info("Both groups have equal return rates")
                    else:
                        st.info("Entry date not available for conversion analysis")

            # Show experiment entries detail
            with st.expander(f"View Experiment Entries ({len(df_experiment_entries):,} total)"):
                df_exp_display = df_experiment_entries.copy()
                if 'entry_date' in df_exp_display.columns:
                    df_exp_display = df_exp_display.sort_values('entry_date', ascending=False)
                st.dataframe(df_exp_display.head(50), use_container_width=True, hide_index=True)
                st.caption("Showing most recent 50 entries")
        else:
            st.info("Experiment group data not available")
    else:
        st.info("No A/B test experiment data available yet. Run the customer flagging pipeline to populate this data.")

# ============================================================================
# TAB 8: EXPENSES (HIDDEN FOR NOW)
# ============================================================================
# with tab8:
#     st.header('Expense Analysis')
#     st.markdown('Payroll and Marketing expenses from QuickBooks (2025 YTD)')
# 
#     if not df_expenses.empty:
#         # Add expense categories
#         df_expenses_categorized = categorize_expenses.add_expense_categories(df_expenses)
# 
#         # Filter to only Payroll and Marketing
#         df_display = df_expenses_categorized[df_expenses_categorized['category_group'].isin(['Payroll', 'Marketing'])].copy()
# 
#         if not df_display.empty:
#             # Ensure date is datetime
#             df_display['date'] = pd.to_datetime(df_display['date'])
# 
#             # Calculate summary metrics
#             total_payroll = df_display[df_display['category_group'] == 'Payroll']['amount'].sum()
#             total_marketing = df_display[df_display['category_group'] == 'Marketing']['amount'].sum()
#             total_expenses = total_payroll + total_marketing
# 
#             # Summary metrics
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric(
#                     "Total Payroll & Marketing",
#                     f"${total_expenses:,.0f}",
#                     help="Combined Payroll and Marketing expenses for 2025 YTD"
#                 )
#             with col2:
#                 st.metric(
#                     "Payroll Expenses",
#                     f"${total_payroll:,.0f}",
#                     help="Salaries, payroll taxes, and employee benefits"
#                 )
#             with col3:
#                 st.metric(
#                     "Marketing Expenses",
#                     f"${total_marketing:,.0f}",
#                     help="Google Ads, social media, website ads, and listing fees"
#                 )
# 
#             # Monthly expense trend
#             st.subheader('Monthly Expense Trends')
# 
#             df_display['year_month'] = df_display['date'].dt.to_period('M').astype(str)
#             monthly_expenses = df_display.groupby(['year_month', 'category_group'])['amount'].sum().reset_index()
# 
#             # Create line chart
#             fig_monthly = go.Figure()
# 
#             for category in ['Payroll', 'Marketing']:
#                 df_cat = monthly_expenses[monthly_expenses['category_group'] == category]
#                 color = COLORS['primary'] if category == 'Payroll' else COLORS['secondary']
# 
#                 fig_monthly.add_trace(go.Scatter(
#                     x=df_cat['year_month'],
#                     y=df_cat['amount'],
#                     name=category,
#                     mode='lines+markers',
#                     line=dict(color=color, width=3),
#                     marker=dict(size=8),
#                     hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>'
#                 ))
# 
#             fig_monthly.update_layout(
#                 plot_bgcolor=COLORS['background'],
#                 paper_bgcolor=COLORS['background'],
#                 font_color=COLORS['text'],
#                 xaxis_title='Month',
#                 yaxis_title='Expense Amount ($)',
#                 hovermode='x unified',
#                 showlegend=True,
#                 legend=dict(
#                     orientation="h",
#                     yanchor="bottom",
#                     y=1.02,
#                     xanchor="right",
#                     x=1
#                 )
#             )
# 
#             st.plotly_chart(fig_monthly, use_container_width=True)
# 
#             # Category breakdown
#             st.subheader('Expense Breakdown')
# 
#             col1, col2 = st.columns(2)
# 
#             with col1:
#                 # Pie chart
#                 summary = categorize_expenses.get_category_summary(df_display)
# 
#                 fig_pie = go.Figure(data=[go.Pie(
#                     labels=summary['category_group'],
#                     values=summary['total_amount'],
#                     marker=dict(colors=[COLORS['primary'], COLORS['secondary']]),
#                     textinfo='label+percent',
#                     hovertemplate='%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>'
#                 )])
# 
#                 fig_pie.update_layout(
#                     plot_bgcolor=COLORS['background'],
#                     paper_bgcolor=COLORS['background'],
#                     font_color=COLORS['text'],
#                     showlegend=False
#                 )
# 
#                 st.plotly_chart(fig_pie, use_container_width=True)
# 
#             with col2:
#                 # Summary table
#                 st.markdown("**Category Summary**")
# 
#                 summary_display = summary.copy()
#                 summary_display['total_amount'] = summary_display['total_amount'].apply(lambda x: f"${x:,.0f}")
#                 summary_display['avg_amount'] = summary_display['avg_amount'].apply(lambda x: f"${x:,.0f}")
#                 summary_display.columns = ['Category', 'Total Amount', 'Transactions', 'Avg Amount']
# 
#                 st.dataframe(summary_display, use_container_width=True, hide_index=True)
# 
#                 # Key insights
#                 st.markdown("**Key Insights:**")
#                 payroll_pct = (total_payroll / total_expenses * 100) if total_expenses > 0 else 0
#                 st.markdown(f"- Payroll represents **{payroll_pct:.1f}%** of tracked expenses")
#                 st.markdown(f"- Average payroll transaction: **${summary[summary['category_group'] == 'Payroll']['avg_amount'].values[0]:,.0f}**")
#                 if len(summary[summary['category_group'] == 'Marketing']) > 0:
#                     st.markdown(f"- Average marketing transaction: **${summary[summary['category_group'] == 'Marketing']['avg_amount'].values[0]:,.0f}**")
# 
#             # Detailed transactions table
#             with st.expander("📋 View Detailed Transactions"):
#                 df_table = df_display[['date', 'category_group', 'expense_category', 'vendor', 'description', 'amount']].copy()
#                 df_table = df_table.sort_values('date', ascending=False)
#                 df_table['date'] = df_table['date'].dt.strftime('%Y-%m-%d')
#                 df_table['amount'] = df_table['amount'].apply(lambda x: f"${x:,.2f}")
#                 df_table.columns = ['Date', 'Category', 'Expense Type', 'Vendor', 'Description', 'Amount']
# 
#                 st.dataframe(df_table, use_container_width=True, hide_index=True, height=400)
# 
#         else:
#             st.info('No Payroll or Marketing expense data available')
# 
#     else:
#         st.info('No expense data available. Run the QuickBooks pipeline to fetch data.')

# Footer
st.markdown('---')
st.caption('Basin Climbing & Fitness Analytics Dashboard | Data updated every 5 minutes')
