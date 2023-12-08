import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.dates as mdates
def get_config():
    """
    Return configurations for data processing and model training.
    The selected features and target variables can be seen in the train-data-all folder
    Innerspeed variables are selected from left top to right bottom of the area
    The target features describe the inner project area of the südiche au.
    """
    return {
        'data_files': [
            "demographics_agegroup.csv",
            "amenities.csv",
            "demographics_area.csv",
            "traffic-data-byquarters.csv",
            "strava-data.csv",
            "weather_data.csv",
            "escooter-counts.csv",
            "emopeds-counts.csv",
            "bike-data.csv"
        ],
        'base_path': "train-data-au/",  # folder for südliche au data
        "grid_search": False,
        'road_closure_date': "12.06.2023",  # closure date of südliche au
        "use_all_features": False,
        "selected_features":
            # [ 'newmobility','percentageclosedstreet',
            #                    'removedparking', 'wpgt_',
            #                   'outer_actmode_',
            #
            #                    'inner_actmode_', 'cultural-inner',  'weekday',
            #                 'outer_speed_', 'tavg_',
            #                    'biketotal_', 'inner_speed1_', 'inner_speed2_',
            #                   'inner_speed3_', 'inner_speed4_', 'inner_speed5_', 'inner_speed6_', 'inner_speed7_',
            #                   'inner_speed8_', 'inner_escooter_', 'outer_escooter_', 'inner_emoped_'],
            #
            [ 'outer_speed_',

             'weekday',
             'inner_speed3_',
             'tmin_'],
        # these are the same features only adding the before name to hinder the model to predict future variables
        # with the same feature values, so for example, inner_speed1 would be predicted by inner_speed1, which makes no sense, so i invent those variables _before, with only the average after the road closure data
        'targets': ['inner_speed'],  # these targets are the inner project area of the süedliche au
        'test_size': 0.2,
        'random_state': 42
    }

# Function to load and merge data
def load_data(data_files, base_path):
    datasets = {}
    for file in data_files:
        key_name = file.split('.')[0]
        datasets[key_name] = pd.read_csv(base_path + file)
    return datasets

def merge_data(datasets):
    merged_data = datasets["traffic-data-byquarters"].copy()
    for key, df in datasets.items():
        if key != "traffic-data-byquarters":
            merged_data = pd.merge(merged_data, df, on="date", how="outer")
    return merged_data

def add_pre_closure_means_by_weekday(merged_data, features, road_closure_date_str):
    merged_data['date'] = pd.to_datetime(merged_data['date'], format='%d.%m.%Y')
    road_closure_date = pd.to_datetime(road_closure_date_str, format='%d.%m.%Y')

    for feature in features:
        new_col_name = f"{feature}_"
        merged_data[new_col_name] = None

        # Copy values before the closure and calculate means based on 'weekday'
        merged_data.loc[merged_data['date'] <= road_closure_date, f"{feature}_"] = merged_data.loc[
            merged_data['date'] <= road_closure_date, feature]

        means = merged_data.loc[merged_data['date'] <= road_closure_date].groupby('weekday')[feature].mean()

        for i, row in merged_data.loc[merged_data['date'] > road_closure_date].iterrows():
            weekday = row['weekday']
            merged_data.at[i, f"{feature}_"] = means[weekday]

    return merged_data

def convert_to_numeric(df, columns):
    """Convert specified columns in dataframe to numeric data type."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def impute_missing_values(df, strategy='mean'):
    """Impute missing values in the dataframe."""
    imputer = SimpleImputer(strategy=strategy)
    columns = df.columns
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=columns)
    return df_imputed
def save_to_csv(df, file_name):
    """
    Save the dataframe to a CSV file.

    Parameters:
    df (pandas.DataFrame): The dataframe to be saved.
    file_name (str): The file name or path where the CSV file will be saved.
    """
    df.to_csv(file_name, index=False)
    print(f"Data saved to {file_name}")
def main():
    config = get_config()
    datasets = load_data(config['data_files'], config['base_path'])
    merged_data = merge_data(datasets)
    features_to_process = ['wspd', 'bikedirection_south',
                           'bikedirection_north', 'tavg', 'inner_speed8',
                           'snow', 'pres', 'wpgt', 'inner_speed1',
                           'outer_actmode', 'inner_speed7',
                           'inner_speed4', 'prcp', 'inner_speed3', 'tmax',
                           'inner_actmode', 'inner_speed5', 'inner_speed6', 'tmin', 'tsun', 'wdir',
                           'inner_speed2', 'outer_speed',
                           'biketotal', "inner_escooter", "outer_escooter", "inner_emoped", "outer_emoped"]
    merged_data = add_pre_closure_means_by_weekday(merged_data, features_to_process, config['road_closure_date'])
    features = config ['selected_features']

    # Convert to numeric and handle missing values
    date_column = merged_data.pop('date')

    # Convert to numeric and handle missing values
    merged_data = convert_to_numeric(merged_data, config['selected_features'] + config['targets'])
    merged_data = impute_missing_values(merged_data)

    # Reattach the 'date' column
    merged_data['date'] = date_column
    save_to_csv(merged_data, 'merged_data.csv')
    # Create a correlation matrix
    corr_matrix = merged_data[features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.show()

    # Preparing data for model
    X = merged_data[features]
    y = merged_data[config['targets']]  # Assuming the first target variable

    # Filter data to only include dates after the road closure date
    road_closure_date = pd.to_datetime(config['road_closure_date'], format='%d.%m.%Y')
    post_closure_filter = merged_data['date'] > road_closure_date
    X_post_closure = X[post_closure_filter]
    y_post_closure = y[post_closure_filter]

    # Impute missing values in post-closure data
    #X_post_closure = impute_missing_values(X_post_closure, numeric_columns)
    #y_post_closure = impute_missing_values(pd.DataFrame(y_post_closure), [config['targets'][0]])

    # Train-test split (using all data for training, post-closure data for testing)
    X_train, X_test, y_train, y_test = X, X_post_closure, y, y_post_closure[config['targets']]

    # Building the linear regression model
    X_train_sm = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_sm).fit()

    # Displaying the model summary
    print(model.summary())

    # Predictions and model evaluation
    X_test_sm = sm.add_constant(X_test)
    y_pred = model.predict(X_test_sm)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")
    # Group by date and calculate mean values
    merged_data['date'] = pd.to_datetime(merged_data['date'])
    mean_values = merged_data.groupby(pd.Grouper(key='date', freq='D')).mean()

    features_for_prediction = mean_values[features]

    # Add a constant to the features (if your model was trained with a constant)
    features_for_prediction = sm.add_constant(features_for_prediction)

    # Flatten the mean_predicted_values to 1D array
    mean_predicted_values = model.predict(features_for_prediction).ravel()

    # Creating a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Date': mean_values.index,
        'Mean True Value': mean_values[config['targets']].values.ravel(),
        'Mean Predicted Value': mean_predicted_values
    })

    # Creating subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    tum_blue = '#072140'
    tum_lighter_blue = '#5E94D4'

    # Filter the DataFrame to include only data from June 12, 2023, onwards
    start_date = pd.to_datetime("2023-06-12")
    plot_data_filtered = plot_data[plot_data['Date'] >= start_date]

    # Determine the maximum value for setting y-axis limits
    max_value = max(plot_data_filtered['Mean True Value'].max(), plot_data_filtered['Mean Predicted Value'].max())

    # Creating subplots
    max_value = max(plot_data_filtered['Mean True Value'].max(), plot_data_filtered['Mean Predicted Value'].max())

    # Creating subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Time-series plot on the first subplot
    axes[0].plot(plot_data_filtered['Date'], plot_data_filtered['Mean True Value'], label='Mean True Value',
                 color=tum_blue)
    axes[0].plot(plot_data_filtered['Date'], plot_data_filtered['Mean Predicted Value'], label='Mean Predicted Value',
                 color=tum_lighter_blue)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Values')
    axes[0].set_title('Mean true vs mean predicted relative speed with Linear Regression:  ' + r'$\bf{Südliche\ Au}$')
    axes[0].set_ylim([0.55, max_value])

    # Setting date format and locator for the first subplot
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())

    axes[0].legend()

    # Scatter plot with a perfect linear line
    axes[1].scatter(plot_data_filtered['Mean True Value'], plot_data_filtered['Mean Predicted Value'], color=tum_lighter_blue, alpha = 0.5)
    axes[1].plot([0.55, max_value], [0.55, max_value], color=tum_blue, linestyle='--')  # perfect linear line
    axes[1].set_xlabel('Mean True Value')
    axes[1].set_ylabel('Mean Predicted Value')
    axes[1].set_title('Scatter plot with Linear Regression for  ' + r'$\bf{Südliche\ Au}$')
    axes[1].set_xlim([0.55, max_value])
    axes[1].set_ylim([0.55, max_value])
    # Adjust layout
    plt.tight_layout()
    plt.show()
    # Assuming you have your model fit in 'model'
    results = model.summary()

    # Convert summary to a pandas dataframe
    results_as_html = results.tables[1].as_html()
    results_df = pd.read_html(results_as_html, header=0, index_col=0)[0]

    # Function to apply stars based on p-values
    def apply_stars(row):
        if row['P>|t|'] < 0.001:
            return row.name + ' ***'
        elif row['P>|t|'] < 0.01:
            return row.name + ' **'
        elif row['P>|t|'] < 0.05:
            return row.name + ' *'
        else:
            return row.name

    # Apply stars to the index (feature names)
    results_df.index = results_df.apply(apply_stars, axis=1)

    # Print the modified DataFrame
    print(results_df)


if __name__ == "__main__":
    main()