import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the predictions data
predictions = pd.read_csv('predictions.csv')

# Convert the date column to datetime
predictions['date'] = pd.to_datetime(predictions['date'])

# Sort the DataFrame based on the 'date' column
predictions.sort_values('date', inplace=True)

# Identify all the prediction columns by filtering the column names
true_columns = [col for col in predictions.columns if col.startswith('True_')]
predicted_columns = [col for col in predictions.columns if col.startswith('Predicted_')]

# Calculate the mean for each row across the true and predicted columns
predictions['Mean_True'] = predictions[true_columns].mean(axis=1)
predictions['Mean_Predicted'] = predictions[predicted_columns].mean(axis=1)

# TUM Colors (RGB)
tum_blue = '#072140'
tum_lighter_blue = '#5E94D4'

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(predictions['date'], predictions['Mean_True'], color=tum_blue, label='True mean relative speed', marker='o')
plt.plot(predictions['date'], predictions['Mean_Predicted'], color=tum_lighter_blue, label='Predicted mean relative speed', marker='x')

# Set x-axis major formatter to show only month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# For better readability, set the major locator to month
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Rotate and align the tick labels so they look better
#plt.gcf().autofmt_xdate()

# Adding a title
plt.title('True and predicted mean relative speed values for Südliche Au')

# Show grid
plt.grid(True)

# Adding the legend
plt.legend()

# Show the plot
plt.show()
