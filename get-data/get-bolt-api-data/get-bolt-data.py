import requests
import uuid
import datetime

# Generate UUID for deviceId
device_id = str(uuid.uuid4())

# API parameters
params = {
    "lat": "48.122815",
    "lng": "11.575976",
    "version": "CI.24.0",
    "deviceId": device_id,
    "deviceType": "iphone",
    "device_name": "iPhone12,3",
    "device_os_version": "iOS15.0",
    "language": "en"
}

# API request
url = "https://rental-search.bolt.eu/categoriesOverview"
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Get current date and time
    now = datetime.datetime.now()
    current_datetime = now.strftime("%Y-%m-%d_%H-%M")

    # Save response data
    output_filename = f"output/bolt_data_{current_datetime}.json"
    with open(output_filename, "w") as file:
        file.write(response.text)
    print("Data saved to:", output_filename)
else:
    print("Error occurred during API request:", response.status_code, response.text)
