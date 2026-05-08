import csv
import sys
import json
from collections import defaultdict
from tqdm import tqdm  # Progress bar library
import os  # For getting the current folder path

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

# Target countries list
target_countries = [
    "加拿大"
]

# Dictionary to store panoIDs and coordinates for each country
country_data = defaultdict(list)

# Input and output file paths
input_file = 'tuxun_combined.csv'  # Replace with your CSV file path
output_file = os.path.join(os.getcwd(), 'panoids_with_coords.csv')  # Save to the current folder

# Initialize remaining countries counter
remaining_countries = set(target_countries)

# Read CSV file and extract panoIDs and coordinates
with open(input_file, 'r', encoding='utf-8') as csvfile:
    # Get file line count to initialize the progress bar
    total_lines = sum(1 for _ in csvfile)
    csvfile.seek(0)  # Reset file pointer

    reader = csv.DictReader(csvfile)
    first_match_found = False  # To flag if the first match is printed

    with tqdm(total=total_lines - 1, desc="Processing", unit="lines") as pbar:
        for row in reader:
            # Parse 'data' field as JSON
            try:
                data = json.loads(row.get('data'))  # Replace 'data' with the actual field name
                rounds = data.get('rounds', [])  # Get 'rounds' list from JSON
            except (json.JSONDecodeError, KeyError, TypeError):
                # Skip row if parsing fails
                pbar.update(1)
                continue

            # Iterate through rounds and extract required information
            for round_data in rounds:
                # Ensure round_data is a valid dictionary
                if not isinstance(round_data, dict):
                    continue

                try:
                    country = round_data.get('nation')  # Get country
                    panoid = round_data.get('panoId')  # Get panoID
                    lat = round_data.get('lat')  # Get latitude
                    lng = round_data.get('lng')  # Get longitude
                except AttributeError:
                    continue

                # Print the first matching result
                if not first_match_found and country in target_countries:
                    print(f"First match: Country: {country}, panoID: {panoid}, lat: {lat}, lng: {lng}")
                    first_match_found = True

                # Check if the country is a target and if fewer than 160 panoIDs are collected
                if country in target_countries and len(country_data[country]) < 160:
                    country_data[country].append((panoid, lat, lng))
                    
                    # If 160 entries are found, remove the country from the remaining list
                    if len(country_data[country]) == 160:
                        print(f"160 panoIDs for {country} have been collected.")
                        remaining_countries.discard(country)

                # Exit if all target countries are complete
                if not remaining_countries:
                    break
            # Exit if all target countries are complete
            if not remaining_countries:
                break
            pbar.update(1)

# Write results to a new CSV file
with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['nation', 'panoID', 'lat', 'lng'])  # Write header
    for country, entries in country_data.items():
        for panoid, lat, lng in entries:
            writer.writerow([country, panoid, lat, lng])

print(f"All collected panoIDs and coordinates have been written to {output_file}")
