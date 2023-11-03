import os
import json

# Specify the directory where the JSON files are located
directory = "output"

# Get the list of JSON files in the directory
json_files = [file for file in os.listdir(directory) if file.endswith(".json")]

# Create a dictionary to store the combined data for each day
combined_data = {}

# Iterate over the JSON files
for file_name in json_files:
    # Extract the date from the file name
    date = file_name.split("_")[2]

    # Load the data from the current file
    with open(os.path.join(directory, file_name), "r") as file:
        data = json.load(file)

    # Check if the date is already present in the combined data
    if date in combined_data:
        # Append the data from the current file to the existing list
        combined_data[date].append(data)
    else:
        # Initialize the combined data for the current date with a list containing the data
        combined_data[date] = [data]

# Create the "output" folder if it doesn't exist
os.makedirs("output", exist_ok=True)

# Save the combined data to individual JSON files for each day
for date, data_list in combined_data.items():
    output_file = os.path.join("output2", f"combined_data_{date}.json")

    # Merge the dictionaries in the data list into a single dictionary
    combined_dict = {}
    for data in data_list:
        combined_dict.update(data)

    with open(output_file, "w") as file:
        json.dump(combined_dict, file, indent=4)

    print(f"Combined data for {date} saved to: {output_file}")
