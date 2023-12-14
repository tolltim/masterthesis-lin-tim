import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_predictions(csv_file1, csv_file2):
    # Read and preprocess both CSV files
    def preprocess(csv_file):
        predictions = pd.read_csv(csv_file)
        predictions['date'] = pd.to_datetime(predictions['date'])
        predictions.sort_values('date', inplace=True)

        true_columns = [col for col in predictions.columns if col.startswith('True_')]
        predicted_columns = [col for col in predictions.columns if col.startswith('Predicted_')]

        predictions['Mean_True'] = predictions[true_columns].mean(axis=1)
        predictions['Mean_Predicted'] = predictions[predicted_columns].mean(axis=1)

        return predictions

    predictions1 = preprocess(csv_file1)
    predictions2 = preprocess(csv_file2)

    # Colors
    tum_blue = '#3070B3'
    tum_dark_blue = '#072140'
    tum_lighter_blue = '#5E94D4'
    tum_orange = '#F7811E'
    tum_green = '#9FBA36'
    black = '#000000'
    tum_gray = '#6A757E'
    tum_light_gray = '#dad7cb'

    # Set up the matplotlib figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # First CSV file: Time series
    axes[0, 0].plot(predictions1['date'], predictions1['Mean_True'], color=tum_blue, label='True mean values (Südliche Au)', marker='o')
    axes[0, 0].plot(predictions1['date'], predictions1['Mean_Predicted'], color=tum_lighter_blue, label='Predicted mean values (Südliche Au)', marker='x')
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    axes[0, 0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0, 0].set_title('Time series true vs. predicted values: ' + r'$\bf{Südliche\ Au}$')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # Second CSV file: Time series
    axes[0, 1].plot(predictions2['date'], predictions2['Mean_True'], color=tum_blue, label='True mean values (Walchenseeplatz)', marker='o')
    axes[0, 1].plot(predictions2['date'], predictions2['Mean_Predicted'], color=tum_lighter_blue, label='Predicted mean values (Walchenseeplatz)', marker='x')
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    axes[0, 1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0, 1].set_title('Time series true vs. predicted relative speeds: ' + r'$\bf{Walchenseeplatz}$')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # First CSV file: Scatter plot
    axes[1, 0].scatter(predictions1['Mean_True'], predictions1['Mean_Predicted'], alpha=0.5, color=tum_lighter_blue)
    axes[1, 0].plot([predictions1['Mean_True'].min(), predictions1['Mean_True'].max()], [predictions1['Mean_True'].min(), predictions1['Mean_True'].max()], '--k')
    axes[1, 0].set_xlabel('True mean values (Südliche Au)')
    axes[1, 0].set_ylabel('Predicted mean values (Südliche Au)')
    axes[1, 0].set_title('Scatter plot: ' + r'$\bf{Südliche\ Au}$')
    axes[1, 0].grid(True)

    # Second CSV file: Scatter plot
    axes[1, 1].scatter(predictions2['Mean_True'], predictions2['Mean_Predicted'], alpha=0.5, color=tum_lighter_blue)
    axes[1, 1].plot([predictions2['Mean_True'].min(), predictions2['Mean_True'].max()], [predictions2['Mean_True'].min(), predictions2['Mean_True'].max()], '--k')
    axes[1, 1].set_xlabel('True mean values (Walchenseeplatz)')
    axes[1, 1].set_ylabel('Predicted mean values (Walchenseeplatz)')
    axes[1, 1].set_title('Scatter plot: ' + r'$\bf{Walchenseeplatz}$')
    axes[1, 1].grid(True)

    # Improve spacing between subplots
    plt.tight_layout()

    # Show the figure
    plt.show(block=True)

# Example usage:
plot_predictions('predictions_au.csv', 'predictions_wp.csv')

