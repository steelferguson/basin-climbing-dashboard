import os
import pprint
from src.projections.membership_projections import MembershipProjectionCalculator, pullCapitanMembershipData

def test_charge_calculation():
    """Test the charge calculation functionality."""
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
    print(f"\n=== Testing Charge Calculation ===")
    print(f"Shaun's bi-weekly charge: ${charge:.2f}")
    assert charge > 0, "Charge should be greater than 0"

def test_membership_categorization():
    """Test the membership categorization functionality."""
    print("\n=== Testing Membership Categorization ===")
    
    test_cases = [
        {
            'name': 'Regular Bi-Weekly Solo',
            'data': {
                'interval': 'BWK',
                'name': 'Solo Weekly',
                'is_recurring': True,
                'customer_add_ons': {'1234': []},
                'duration_months': None
            },
            'expected': {'frequency': 'bi_weekly', 'type': 'solo', 'has_fitness': False}
        },
        {
            'name': '3-Month Prepaid',
            'data': {
                'interval': '',
                'name': 'Solo 3 Months Prepaid',
                'is_recurring': False,
                'customer_add_ons': {'1234': []},
                'duration_months': 3
            },
            'expected': {'frequency': 'prepaid_3mo', 'type': 'solo', 'has_fitness': False}
        },
        {
            'name': 'Family with Fitness',
            'data': {
                'interval': 'MON',
                'name': 'Family Monthly',
                'is_recurring': True,
                'customer_add_ons': {'1234': [{'name': 'Unlimited Fitness Add-on', 'fee': '15'}]},
                'duration_months': None
            },
            'expected': {'frequency': 'monthly', 'type': 'family', 'has_fitness': True}
        }
    ]
    
    calculator = MembershipProjectionCalculator()
    for test_case in test_cases:
        result = calculator.categorize_membership(test_case['data'])
        print(f"\nTesting: {test_case['name']}")
        print(f"Expected: {test_case['expected']}")
        print(f"Got:      {result}")
        assert result == test_case['expected'], f"Categorization failed for {test_case['name']}"

def test_unknown_frequencies():
    """Test detection of unknown frequency memberships."""
    print("\n=== Testing Unknown Frequency Detection ===")
    
    # Check for API token
    if not os.getenv('CAPITAN_API_TOKEN'):
        print("Error: CAPITAN_API_TOKEN environment variable not set")
        return

    print("\nFetching memberships data...")
    membership_data = pullCapitanMembershipData.get_memberships()
    
    if not membership_data:
        print("Failed to retrieve membership data.")
        return
        
    print(f"\nFound {len(membership_data.get('results', []))} total memberships")
    
    calculator = MembershipProjectionCalculator()
    unknown_count = 0
    
    print("\nSearching for unknown frequency memberships...")
    for membership in membership_data.get('results', []):
        if membership.get('status') == 'ACT':
            categories = calculator.categorize_membership(membership)
            
            if categories['frequency'] == 'unknown':
                unknown_count += 1
                print("\n" + "="*80)
                print(f"UNKNOWN FREQUENCY MEMBERSHIP #{unknown_count}")
                print("="*80)
                print(f"Name: {membership.get('name')}")
                print(f"Interval: {membership.get('interval')}")
                print(f"Owner: {membership.get('owner_first_name')} {membership.get('owner_last_name')}")
                print(f"Amount: ${float(membership.get('billing_amount', 0)):.2f}")
                print("\nFull API Response:")
                pprint.pprint(membership)
                print("="*80)
    
    print(f"\nFound {unknown_count} memberships with unknown frequency")
    assert unknown_count == 0, f"Found {unknown_count} memberships with unknown frequency"

if __name__ == "__main__":
    # Run all tests
    test_charge_calculation()
    test_membership_categorization()
    test_unknown_frequencies()
    