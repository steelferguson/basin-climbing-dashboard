import requests
import os
import pprint
from membership_projections import MembershipProjectionCalculator, pullCapitanMembershipData

def test_unknown_frequencies():
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

if __name__ == "__main__":
    test_unknown_frequencies() 