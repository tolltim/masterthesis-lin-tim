import pandas as pd
from meteostat import Point, Daily
from datetime import datetime

# Kolumbusplatz
latitude = 48.1227034
longitude = 11.5757801

# Start and end date
start_date = datetime(2023, 5, 2)
end_date = datetime(2023, 9, 21)

point = Point(latitude, longitude)

# Fetch historical weather data
data = Daily(point, start=start_date, end=end_date)
data = data.fetch()

# Convert to pandas
weather_dataframe = data

# Add the 'date' column
weather_dataframe['date'] = weather_dataframe.index

# Prepare CSV file
csv_filename = "weather_data.csv"

# Save the Pd dataframe to CSV
weather_dataframe.to_csv(csv_filename, index=False)

print(f"Weather data saved to {csv_filename}")
