import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Specify the name of the .csv
predictions = pd.read_csv('predictions_wp.csv')

predictions['date'] = pd.to_datetime(predictions['date'])

predictions.sort_values('date', inplace=True)

# Identify all the prediction columns
true_columns = [col for col in predictions.columns if col.startswith('True_')]
predicted_columns = [col for col in predictions.columns if col.startswith('Predicted_')]

# Calculate the mean
predictions['Mean_True'] = predictions[true_columns].mean(axis=1)
predictions['Mean_Predicted'] = predictions[predicted_columns].mean(axis=1)

tum_blue = '#072140'
tum_lighter_blue = '#5E94D4'

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(predictions['date'], predictions['Mean_True'], color=tum_blue, label='True mean relative speed', marker='o')
plt.plot(predictions['date'], predictions['Mean_Predicted'], color=tum_lighter_blue, label='Predicted mean relative speed', marker='x')

# Set x-axis major formatter to show only month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.title('True and predicted mean relative speed values for Walchenseeplatz')
plt.grid(True)
plt.legend()
plt.show()
