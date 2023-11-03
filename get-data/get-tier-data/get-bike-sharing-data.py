import requests
import json
import os
from datetime import datetime
import time

def get_vehicle_data(lat, lng, radius):
    url = f"https://platform.tier-services.io/v1/vehicle?lat={lat}&lng={lng}&radius={radius}"
    ### you need to get the x-api-key by yourself
    headers = {
        "X-Api-Key": ""
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

# Specify the location (latitude and longitude) and radius
latitude = 48.122815
longitude = 11.575976
radius = 10000

vehicle_data = get_vehicle_data(latitude, longitude, radius)
desired_attributes = []
if vehicle_data:
    for vehicle in vehicle_data["data"]:
        attributes = vehicle["attributes"]
        desired_attributes.append({
            "id": vehicle["id"],
            "type": vehicle["type"],
            "lat": attributes["lat"],
            "lng": attributes["lng"],
            "state": attributes["state"],
            "vehicleType": attributes["vehicleType"],
            "zoneID": attributes["zoneId"],
            "batteryLevel": attributes["batteryLevel"],
            "savedDate": datetime.now().strftime("%Y-%m-%d"),
            "savedTime": datetime.now().strftime("%H:%M:%S")
        })
        print(f"id: {vehicle['id']}, type: {vehicle['type']}, lat: {attributes['lat']}, lng: {attributes['lng']}, "
              f"state: {attributes['state']}, vehicleType: {attributes['vehicleType']}, zoneID: {attributes['zoneId']}, "
              f"batteryLevel: {attributes['batteryLevel']}")


os.makedirs("output", exist_ok=True)
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
current_date = datetime.now().strftime("%Y-%m-%d")

# Specify the output file
output_file = f"output/tier_vehicle_data_{current_date}.json"

# Check if the output file already exists for the current date
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        existing_data = json.load(f)

    existing_data.extend(desired_attributes)
    desired_attributes = existing_data

if desired_attributes:
    with open(output_file, "w") as f:
        json.dump(desired_attributes, f, indent=4)
    print(f"\nWurde hier gespeichert: {output_file}")
    time.sleep(2)
else:
    print("Failed to retrieve vehicle data or no nothing found.")
