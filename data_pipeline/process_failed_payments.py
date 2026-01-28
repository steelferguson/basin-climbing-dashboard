"""
Process failed payment data and calculate failure rates by membership type.

This script enriches failed payment data with membership information
to calculate failure rates by membership category (college, founder, etc.)
"""

import pandas as pd
from data_pipeline import config, upload_data


def enrich_failed_payments_with_membership_data(
    df_failed_payments: pd.DataFrame,
    df_memberships: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge failed payments with membership data to add membership type info.

    Returns DataFrame with added columns:
    - is_college, is_founder, is_corporate, etc. (membership flags)
    - size (solo, duo, family, corporate)
    - frequency (monthly, annual, etc.)
    - name (membership type name)
    """
    # Merge on membership_id
    df_enriched = df_failed_payments.merge(
        df_memberships[['membership_id', 'name', 'size', 'frequency',
                        'is_college', 'is_founder', 'is_corporate', 'is_mid_day',
                        'is_fitness_only', 'has_fitness_addon', 'is_team_dues',
                        'is_bcf', 'is_90_for_90', 'is_not_in_special', 'status']],
        on='membership_id',
        how='left'
    )

    return df_enriched


def calculate_failure_rates_by_type(
    df_failed_payments: pd.DataFrame,
    df_memberships: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate payment failure rates by membership type.

    Returns DataFrame with columns:
    - membership_type: Type of membership (college, founder, solo, etc.)
    - active_memberships: Number of active memberships of this type
    - unique_with_failures: Number of unique memberships with failures
    - total_failures: Total number of failed payments
    - insufficient_funds_failures: Number of insufficient_funds failures
    - failure_rate: % of memberships with any failure
    - insufficient_funds_rate: % of memberships with insufficient_funds failure
    """
    # Only consider active memberships for rates
    active_memberships = df_memberships[df_memberships['status'] == 'ACT'].copy()

    results = []

    # Define membership categories to analyze
    categories = {
        'College': 'is_college',
        'Founder': 'is_founder',
        'Corporate': 'is_corporate',
        'Mid-Day': 'is_mid_day',
        'Fitness Only': 'is_fitness_only',
        'Team Dues': 'is_team_dues',
        'BCF Staff/Family': 'is_bcf',
        '90 for 90': 'is_90_for_90',
        'Standard (no special category)': 'is_not_in_special',
    }

    for category_name, flag_column in categories.items():
        # Get active memberships of this type
        category_memberships = active_memberships[active_memberships[flag_column] == True]
        count_active = len(category_memberships)

        if count_active == 0:
            continue

        # Get membership IDs for this category
        category_membership_ids = set(category_memberships['membership_id'].values)

        # Filter failed payments to this category
        category_failures = df_failed_payments[
            df_failed_payments['membership_id'].isin(category_membership_ids)
        ]

        # Calculate metrics
        total_failures = len(category_failures)
        unique_with_failures = category_failures['membership_id'].nunique()

        insufficient_funds = category_failures[
            category_failures['decline_code'] == 'insufficient_funds'
        ]
        insufficient_funds_count = len(insufficient_funds)
        unique_with_insuff_funds = insufficient_funds['membership_id'].nunique()

        failure_rate = (unique_with_failures / count_active * 100) if count_active > 0 else 0
        insuff_funds_rate = (unique_with_insuff_funds / count_active * 100) if count_active > 0 else 0

        results.append({
            'membership_type': category_name,
            'active_memberships': count_active,
            'unique_with_failures': unique_with_failures,
            'total_failures': total_failures,
            'insufficient_funds_failures': insufficient_funds_count,
            'unique_with_insuff_funds': unique_with_insuff_funds,
            'failure_rate_pct': round(failure_rate, 1),
            'insufficient_funds_rate_pct': round(insuff_funds_rate, 1),
        })

    # Also calculate by size (solo, duo, family, corporate)
    for size in active_memberships['size'].unique():
        size_memberships = active_memberships[active_memberships['size'] == size]
        count_active = len(size_memberships)

        if count_active == 0:
            continue

        size_membership_ids = set(size_memberships['membership_id'].values)
        size_failures = df_failed_payments[
            df_failed_payments['membership_id'].isin(size_membership_ids)
        ]

        total_failures = len(size_failures)
        unique_with_failures = size_failures['membership_id'].nunique()

        insufficient_funds = size_failures[
            size_failures['decline_code'] == 'insufficient_funds'
        ]
        insufficient_funds_count = len(insufficient_funds)
        unique_with_insuff_funds = insufficient_funds['membership_id'].nunique()

        failure_rate = (unique_with_failures / count_active * 100) if count_active > 0 else 0
        insuff_funds_rate = (unique_with_insuff_funds / count_active * 100) if count_active > 0 else 0

        results.append({
            'membership_type': f"{size.title()} (by size)",
            'active_memberships': count_active,
            'unique_with_failures': unique_with_failures,
            'total_failures': total_failures,
            'insufficient_funds_failures': insufficient_funds_count,
            'unique_with_insuff_funds': unique_with_insuff_funds,
            'failure_rate_pct': round(failure_rate, 1),
            'insufficient_funds_rate_pct': round(insuff_funds_rate, 1),
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('insufficient_funds_rate_pct', ascending=False)

    return df_results


if __name__ == "__main__":
    # Load data from S3
    uploader = upload_data.DataUploader()

    # Load memberships
    csv_content = uploader.download_from_s3(
        config.aws_bucket_name,
        config.s3_path_capitan_memberships
    )
    df_memberships = uploader.convert_csv_to_df(csv_content)

    # Load failed payments
    csv_content = uploader.download_from_s3(
        config.aws_bucket_name,
        config.s3_path_failed_payments
    )
    df_failed_payments = uploader.convert_csv_to_df(csv_content)

    # Enrich failed payments with membership data
    df_enriched = enrich_failed_payments_with_membership_data(
        df_failed_payments,
        df_memberships
    )

    # Calculate failure rates
    df_rates = calculate_failure_rates_by_type(
        df_failed_payments,
        df_memberships
    )

    print("=" * 80)
    print("PAYMENT FAILURE RATES BY MEMBERSHIP TYPE")
    print("=" * 80)
    print()
    print(df_rates.to_string(index=False))
    print()

    # Save enriched data
    df_enriched.to_csv('data/outputs/failed_payments_enriched.csv', index=False)
    df_rates.to_csv('data/outputs/failure_rates_by_membership_type.csv', index=False)

    print("Results saved to:")
    print("  - data/outputs/failed_payments_enriched.csv")
    print("  - data/outputs/failure_rates_by_membership_type.csv")
