from pullDataFromCapitan import pullDataFromCapitan

def main():
    # Create an instance of the Capitan data puller
    capitan = pullDataFromCapitan()
    
    # Fetch and save memberships data
    print("Fetching memberships data from Capitan API...")
    memberships_df = capitan.fetch_and_save_memberships()
    
    if memberships_df is not None:
        print("Successfully saved memberships data to JSON file")
    else:
        print("Failed to fetch memberships data")

if __name__ == "__main__":
    main() 