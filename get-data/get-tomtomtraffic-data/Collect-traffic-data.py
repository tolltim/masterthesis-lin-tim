import requests
import json
from datetime import datetime
import os
import csv
import time

# Set up the API call with your TomTom API key and segment URL with zoom on level 9 which is equal to around 300 m tile slide
api_key = "" # insert an api key
segment_url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/9/json"

# Define the path to the CSV file with the measurement points
measurement_points_file = "measurement-points.csv"

output_directory = "traffic_data"

# create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# function to call the API and store the results in a dictionary
def call_tomtom_api(name, lat, lon):
    response = requests.get(segment_url, params={"key": api_key, "point": f"{lat},{lon}"})
    data = response.json()

    flow_data = {
        "name": name,
        "latitude": lat,
        "longitude": lon,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "currentspeed": data["flowSegmentData"]["currentSpeed"],
        "freeflowspeed": data["flowSegmentData"]["freeFlowSpeed"],
        "freeflowtraveltime": data["flowSegmentData"]["freeFlowTravelTime"],
        "currentfreeflowtraveltime": data["flowSegmentData"]["currentTravelTime"],
        "roadclosure": data["flowSegmentData"]["roadClosure"],
        "confidence": data["flowSegmentData"]["confidence"],
        "percentage_speed": round((data["flowSegmentData"]["currentSpeed"] / data["flowSegmentData"]["freeFlowSpeed"]) * 100, 2),
        "percentage_flow": round((data["flowSegmentData"]["currentTravelTime"] / data["flowSegmentData"]["freeFlowTravelTime"]) * 100, 2)
    }

    # print the data to the console
    print(f"Name: {flow_data['name']}")
    print(f"Timestamp: {flow_data['timestamp']}")
    print(f"Current Speed: {flow_data['currentspeed']}")
    print(f"Free Flow Speed: {flow_data['freeflowspeed']}")
    print(f"Free Flow Travel Time: {flow_data['freeflowtraveltime']}")
    print(f"Current Free Flow Travel Time: {flow_data['currentfreeflowtraveltime']}")
    print(f"Road Closure: {flow_data['roadclosure']}")
    print(f"Confidence: {flow_data['confidence']}")
    print(f"Percentage_flow: {flow_data['percentage_flow']}")
    print(f"Percentage_speed: {flow_data['percentage_speed']}\n")

    return flow_data


# function to write the flow data to a JSON file
def write_to_json(data):
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_file = os.path.join(output_directory, f"traffic_flow_data_{current_date}.json")
    with open(output_file, "a") as f:
        json_entry = {
            "name": data["name"],
            "latitude": data["latitude"],
            "longitude": data["longitude"],
            "timestamp": data["timestamp"],
            "currentspeed": data["currentspeed"],
            "freeflowspeed": data["freeflowspeed"],
            "freeflowtraveltime": data["freeflowtraveltime"],
            "currentfreeflowtraveltime": data["currentfreeflowtraveltime"],
            "roadclosure": data["roadclosure"],
            "confidence": data["confidence"],
            "percentage_speed": data["percentage_speed"],
            "percentage_flow": data["percentage_flow"]
        }
        f.write(json.dumps(json_entry) + "\n")



# main function to run the script
def main():
    with open(measurement_points_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip header row
        for row in reader:
            name, lat, lon = row
            flow_data = call_tomtom_api(name, float(lat), float(lon))
            write_to_json(flow_data)

if __name__ == "__main__":
    main()
    print("Good Job, it seems like everything works you idiot sandwich")
    time.sleep(3)

