from src.membership_projections import MembershipProjectionCalculator

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
    