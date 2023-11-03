import os
import json


directory = "output"

# Get the list of JSON files in the directory
json_files = [file for file in os.listdir(directory) if file.endswith(".json")]

# Create a dictionary
combined_data = {}

# Iterate over the JSON files
for file_name in json_files:
    date = file_name.split("_")[2]

    with open(os.path.join(directory, file_name), "r") as file:
        data = json.load(file)

    if date in combined_data:
        # Append the data
        combined_data[date].append(data)
    else:
        # Initialize the combined data
        combined_data[date] = [data]

os.makedirs("output", exist_ok=True)

for date, data_list in combined_data.items():
    output_file = os.path.join("output2", f"combined_data_{date}.json")

    # Merge the dictionaries
    combined_dict = {}
    for data in data_list:
        combined_dict.update(data)

    with open(output_file, "w") as file:
        json.dump(combined_dict, file, indent=4)

    print(f"Combined data for {date} saved to: {output_file}")
