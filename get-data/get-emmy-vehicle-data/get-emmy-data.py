import requests
import json
import datetime
import time
import os

# API endpoint URL
url = "https://emmy.frontend.fleetbird.eu/api/prod/v1.06/map/cars/"


def save_json_data(data):
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    hour = now.strftime("%H-%M")
    output_folder = "output"
    output_file = f"{output_folder}/emmy_data_{date}.json"

    # Create the output folder
    os.makedirs(output_folder, exist_ok=True)

    # Check if the output file already exists for the current date
    if os.path.exists(output_file):
        # Load json data existing
        with open(output_file, "r") as file:
            existing_data = json.load(file)

        # Update the existing data with the current hour's data
        existing_data["hourly_data"][hour] = data["vehicles"]
        data = existing_data

    # Prepare the output JSON structure
    output = {
        "date": date,
        "hourly_data": {
            hour: data["vehicles"]
        }
    }

    # Save the output to a JSON file
    with open(output_file, "w") as file:
        json.dump(output, file, indent=4)

    print(f"Daten wurden hier gespeichert: {output_file}")
    print("Output:")
    print(json.dumps(output, indent=4))
    time.sleep(2)


response = requests.get(url)


if response.status_code == 200:
    try:

        data = response.json()

        # Filter the vehicles for Munich (city = "München") and extract desired fields
        munich_vehicles = [
            {
                "carId": vehicle["carId"],
                "lat": vehicle["lat"],
                "lon": vehicle["lon"],
                "licencePlate": vehicle["licencePlate"],
                "fuelLevel": vehicle["fuelLevel"]
            }
            for vehicle in data if vehicle.get("city") == "München" and vehicle.get("carId") is not None
        ]


        now = datetime.datetime.now()
        output_data = {
            "date": now.strftime("%Y-%m-%d"),
            "hour": now.strftime("%H:%M"),
            "vehicles": munich_vehicles
        }


        save_json_data(output_data)

    except json.JSONDecodeError as e:
        print(f"Error owhile parsing JSON: {e}")
else:
    print(f"Error while API request: {response.status_code} {response.reason}")
