import pandas as pd
import requests
import json
from datetime import timedelta, datetime
import os 
import pprint

class MembershipProjectionCalculator:
    @staticmethod
    def calculate_charge(membership_data):
        """Calculate the total charge for a membership including add-ons, discounts, and tax."""
        # Get base amount
        base_amount = float(membership_data['billing_amount'])
        
        # Calculate add-ons
        add_ons = 0
        for customer_id, add_on_list in membership_data['customer_add_ons'].items():
            for add_on in add_on_list:
                add_ons += float(add_on['fee'])
        
        # Calculate subtotal before discount
        subtotal = base_amount + add_ons
        
        # Apply recurring discount if exists
        if membership_data.get('recurring_discount_amount'):
            discount_percentage = float(membership_data['recurring_discount_amount']) / 100
            discount_amount = base_amount * discount_percentage
            subtotal -= discount_amount
        
        # Apply tax (8.25%)
        tax = subtotal * 0.0825
        
        return subtotal + tax

    @staticmethod
    def categorize_membership(membership_data):
        """Categorize membership by frequency and type."""
        # Frequency categorization
        frequency_map = {
            'BWK': 'bi_weekly',
            'MON': 'monthly',
            'YRL': 'yearly'
        }
        
        # Type categorization based on name
        name = membership_data['name'].lower()
        if 'solo' in name:
            membership_type = 'solo'
        elif 'duo' in name:
            membership_type = 'duo'
        elif 'family' in name:
            membership_type = 'family'
        else:
            membership_type = 'other'
            
        # Check for fitness add-on
        has_fitness = any(
            add_on['name'] == 'Unlimited Fitness Add-on'
            for add_ons in membership_data['customer_add_ons'].values()
            for add_on in add_ons
        )
        
        return {
            'frequency': frequency_map.get(membership_data['interval'], 'unknown'),
            'type': membership_type,
            'has_fitness': has_fitness
        }

    @staticmethod
    def create_projection(membership_data):
        """Create a 3-month projection of income from a membership."""
        if membership_data['status'] != 'ACT':
            return None
            
        charge = MembershipProjectionCalculator.calculate_charge(membership_data)
        categories = MembershipProjectionCalculator.categorize_membership(membership_data)
        
        # Get upcoming bill dates
        bill_dates = membership_data['upcoming_bill_dates']
        
        # Create projection dictionary
        projection = {}
        for date in bill_dates:
            projection[date] = {
                'amount': charge,
                'categories': categories,
                'customer': f"{membership_data['owner_first_name']} {membership_data['owner_last_name']}"
            }
            
        return projection

class pullCapitanMembershipData:
    @staticmethod
    def get_base_and_headers(type='customer-memberships'):
        # 'members'
        my_token = os.getenv('CAPITAN_API_TOKEN')

        if not my_token:
            raise ValueError("API token not found. Please set CAPITAN_API_TOKEN as an environment variable.")

        ## Make the API call
        ## Set up URLs
        url_base = 'https://api.hellocapitan.com/api/'
        url_memberships =  url_base + type + '/' + '?page=1&page_size=10000000000'

        ## Set up headers
        headers={'Authorization': 'token {}'.format(my_token)}
        return url_memberships, headers

    @staticmethod
    def get_memberships():
        url_memberships, headers = pullCapitanMembershipData.get_base_and_headers()
        response = requests.get(url_memberships, headers=headers)
        if response.status_code == 200:
            print("Successfully retrieved memberships data.")
            return response.json()  # Returns the JSON response containing membership data
        else:
            print(f"Failed to retrieve memberships data. Status code: {response.status_code}")
            return None

    @staticmethod
    def create_comprehensive_projection():
        """Create a comprehensive projection for all active memberships."""
        membership_data = pullCapitanMembershipData.get_memberships()
        if not membership_data:
            return None
            
        calculator = MembershipProjectionCalculator()
        all_projections = {}
        membership_summary = {
            'total_memberships': 0,
            'by_frequency': {'bi_weekly': 0, 'monthly': 0, 'yearly': 0, 'unknown': 0},
            'by_type': {'solo': 0, 'duo': 0, 'family': 0, 'other': 0},
            'with_fitness': 0
        }
        
        # Process each membership
        for membership in membership_data.get('results', []):
            if membership.get('status') == 'ACT':
                categories = calculator.categorize_membership(membership)
                membership_summary['total_memberships'] += 1
                membership_summary['by_frequency'][categories['frequency']] += 1
                membership_summary['by_type'][categories['type']] += 1
                if categories['has_fitness']:
                    membership_summary['with_fitness'] += 1
                
                projection = calculator.create_projection(membership)
                if projection:
                    # Merge projections by date
                    for date, details in projection.items():
                        if date not in all_projections:
                            all_projections[date] = []
                        all_projections[date].append(details)
        
        # Sort by date
        return dict(sorted(all_projections.items())), membership_summary

# Example usage
if __name__ == "__main__":
    # Test with Shaun's data
    test_data = {
        'billing_amount': '36.40',
        'customer_add_ons': {'1843837': [{'fee': '4', 'name': 'Gear Upgrade'}]},
        'recurring_discount_amount': '10.000',
        'status': 'ACT',
        'interval': 'BWK',
        'name': 'Solo Weekly',
        'owner_first_name': 'Shaun',
        'owner_last_name': 'Bruner',
        'upcoming_bill_dates': ['2025-04-05', '2025-04-19']
    }
    
    calculator = MembershipProjectionCalculator()
    charge = calculator.calculate_charge(test_data)
    print(f"Shaun's bi-weekly charge: ${charge:.2f}")
    
    # Create comprehensive projection
    projections, membership_summary = pullCapitanMembershipData.create_comprehensive_projection()
    if projections:
        print("\n=== Membership Summary ===")
        print(f"Total Active Memberships: {membership_summary['total_memberships']}")
        print("\nBy Frequency:")
        for freq, count in membership_summary['by_frequency'].items():
            print(f"  {freq.replace('_', ' ').title()}: {count}")
        print("\nBy Type:")
        for type_, count in membership_summary['by_type'].items():
            print(f"  {type_.title()}: {count}")
        print(f"\nWith Fitness Add-on: {membership_summary['with_fitness']}")
        print("========================\n")
        
        print("\n=== Projected Income (Next 3 Months) ===")
        total_projected = 0
        current_date = datetime.now()
        three_months_later = current_date + timedelta(days=90)
        
        # Prepare data for CSV
        csv_data = []
        
        # Calculate totals by category
        total_by_frequency = {}
        total_by_type = {}
        total_by_fitness = {'with_fitness': 0, 'without_fitness': 0}
        
        for date, charges in projections.items():
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            if current_date <= date_obj <= three_months_later:
                total = sum(charge['amount'] for charge in charges)
                total_projected += total
                
                # Group charges by category for this date
                by_frequency = {}
                by_type = {}
                by_fitness = {'with_fitness': 0, 'without_fitness': 0}
                customers = []
                
                for charge in charges:
                    freq = charge['categories']['frequency']
                    type_ = charge['categories']['type']
                    has_fitness = charge['categories']['has_fitness']
                    customers.append(charge['customer'])
                    
                    by_frequency[freq] = by_frequency.get(freq, 0) + charge['amount']
                    by_type[type_] = by_type.get(type_, 0) + charge['amount']
                    if has_fitness:
                        by_fitness['with_fitness'] += charge['amount']
                    else:
                        by_fitness['without_fitness'] += charge['amount']
                    
                    # Add to totals
                    total_by_frequency[freq] = total_by_frequency.get(freq, 0) + charge['amount']
                    total_by_type[type_] = total_by_type.get(type_, 0) + charge['amount']
                    if has_fitness:
                        total_by_fitness['with_fitness'] += charge['amount']
                    else:
                        total_by_fitness['without_fitness'] += charge['amount']
                
                # Add row to CSV data
                csv_row = {
                    'Date': date,
                    'Total': total,
                    'Customers': '; '.join(customers),
                    'Bi Weekly': by_frequency.get('bi_weekly', 0),
                    'Monthly': by_frequency.get('monthly', 0),
                    'Yearly': by_frequency.get('yearly', 0),
                    'Unknown Frequency': by_frequency.get('unknown', 0),
                    'Solo': by_type.get('solo', 0),
                    'Duo': by_type.get('duo', 0),
                    'Family': by_type.get('family', 0),
                    'Other': by_type.get('other', 0),
                    'With Fitness': by_fitness['with_fitness'],
                    'Without Fitness': by_fitness['without_fitness']
                }
                csv_data.append(csv_row)
                
                print(f"\n{date}: ${total:.2f}")
                print("  By Frequency:")
                for freq, amount in by_frequency.items():
                    print(f"    {freq.replace('_', ' ').title()}: ${amount:.2f}")
                print("  By Type:")
                for type_, amount in by_type.items():
                    print(f"    {type_.title()}: ${amount:.2f}")
                print("  By Fitness Add-on:")
                print(f"    With Fitness: ${by_fitness['with_fitness']:.2f}")
                print(f"    Without Fitness: ${by_fitness['without_fitness']:.2f}")
        
        print("\n=== 3-Month Totals by Category ===")
        print("By Frequency:")
        for freq, amount in total_by_frequency.items():
            print(f"  {freq.replace('_', ' ').title()}: ${amount:.2f}")
        print("\nBy Type:")
        for type_, amount in total_by_type.items():
            print(f"  {type_.title()}: ${amount:.2f}")
        print("\nBy Fitness Add-on:")
        print(f"  With Fitness: ${total_by_fitness['with_fitness']:.2f}")
        print(f"  Without Fitness: ${total_by_fitness['without_fitness']:.2f}")
        print(f"\nTotal projected income (next 3 months): ${total_projected:.2f}")
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        csv_filename = f"membership_projections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\nProjections saved to {csv_filename}")