import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.ion()


def get_config():
    """
    Return configurations for data processing and model training.
    The selected features and target variables can be seen in the speedmeasurement-points.png
    Here proposed selected features are built upon on training and testing
    The target features describe the inner project area of the südiche au.
    see picture to change the variables
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
        'base_path': "..train-data-all/",
        "grid_search": False,
        'road_closure_date': "12.06.2023",
        "use_all_features": False,
        "selected_features": ["weekday", "86_age", "87_age", "88_age", "89_age", "90_age",
                              "91_age", "92_age", "1_age", "2_age", "3_age", "4_age", "5_age", "6_age", "7_age",
                              "8_age",
                              "9_age", "10_age", "11_age", "12_age", "13_age", "14_age", "15_age", "16_age",
                              "17_age", "18_age",
                              "19_age", "20_age", "21_age", "22_age", "23_age", "24_age", "25_age", "26_age",
                              "27_age", "28_age","84_before", "85_before", "86_before",
                               "87_before", "88_before","89_before", "90_before", "91_before", "92_before",
                              "1_before", "2_before", "3_before", "4_before", "5_before", "6_before", "7_before",
                              "8_before", "9_before", "10_before",
                              "11_before", "12_before", "13_before", "14_before", "15_before", "16_before", "17_before",
                              "18_before",  "21_before", "22_before", "23_before", "24_before", "25_before", "26_before", "27_before",
                              "31_before", "32_before", "33_before", "34_before", "35_before", "36_before", "37_before",
                              "41_before", "42_before", "43_before", "44_before", "45_before", "46_before", "47_before",
                              "48_before",  "52_before", "53_before", "54_before", "55_before", "56_before", "57_before",
                              "58_before",  "62_before", "63_before", "64_before", "65_before", "66_before", "67_before", "72_before", "73_before",
                              "removedparking-kol",  "1_emoped_", "2_emoped_",
                              "4_emoped_", "5_emoped_", "8_emoped_",
                              "15_emoped_", "87_emoped_", "88_emoped_",
                              "89_emoped_", "90_emoped_", "92_emoped_", "1_escooter_",
                              "2_escooter_", "4_escooter_", "5_escooter_", "8_escooter_",
                              "15_escooter_",
                              "87_escooter_", "88_escooter_", "89_escooter_", "90_escooter_",
                              "92_escooter_", "tavg_before",
                              "tmin_before", "tmax_before", "prcp_before", "snow_before", "wdir_before", "wspd_before",
                              "wpgt_before", "pres_before", "tsun_before","education-kol-inner", "education-landl-inner", "education-kol-outer",
                                 "education-landl-outer",
                                 "consumption-kol-inner", "consumption-landl-inner", "consumption-kol-outer",
                                 "consumption-landl-outer",
                                 "transportation-kol-inner", "transportation-landl-inner", "transportation-kol-outer",
                                 "transportation-landl-outer",
                                 "cultural-kol-inner", "cultural-landl-inner", "cultural-kol-outer",
                                 "cultural-landl-outer",
                                 "bicycle-kol-inner", "bicycle-landl-inner", "bicycle-kol-outer", "bicycle-landl-outer",
                              ],
        'targets': ['87','89', '90', '15', '34', '24', '92', '45'],  # these targets are the inner porject area of the süedliche au
        'test_size': 0.2,
        'random_state': 42
    }


def load_data(data_files, base_path):
    """
    Load datasets from provided files.
    """
    datasets = {}
    for file in data_files:
        key_name = file.split('.')[0]
        datasets[key_name] = pd.read_csv(base_path + file)
    return datasets


def add_pre_closure_means_by_weekday(merged_data, features, road_closure_date_str):
    # Ensure 'date' is in the correct format (do not set as index yet)
    merged_data['date'] = pd.to_datetime(merged_data['date'], format='%d.%m.%Y')

    # Convert the road_closure_date from string to datetime
    road_closure_date = pd.to_datetime(road_closure_date_str, format='%d.%m.%Y')

    # Initialize the new columns
    for feature in features:
        new_col_name = f"{feature}_before"
        merged_data[new_col_name] = None  # Initialize the columns with None

    # Copy values before the road closure and calculate means based on 'weekday'
    for feature in features:
        # Copy the values before the closure date
        merged_data.loc[merged_data['date'] <= road_closure_date, f"{feature}_before"] = merged_data.loc[merged_data['date'] <= road_closure_date, feature]

        # Calculate mean for each weekday before the road closure
        means = merged_data.loc[merged_data['date'] <= road_closure_date].groupby('weekday')[feature].mean()

        # Assign the mean values based on weekday for the dates after the road closure
        for i, row in merged_data.loc[merged_data['date'] > road_closure_date].iterrows():
            weekday = row['weekday']
            merged_data.at[i, f"{feature}_before"] = means[weekday]

    # Return the updated DataFrame
    return merged_data


def merge_data(datasets):
    """
    Merge datasets based on date.
    """
    merged_data = datasets["traffic-data-byquarters"].copy()
    for key, df in datasets.items():
        if key != "traffic-data-byquarters":
            merged_data = pd.merge(merged_data, df, on="date", how="outer")
    return merged_data


def preprocess_for_test(merged_data, road_closure_date):
    """
    Preprocess data to get data after the road closure date for testing.
    """
    merged_data["date"] = pd.to_datetime(merged_data["date"], format="%d.%m.%Y")
    merged_data_after = merged_data[
        merged_data["date"] > pd.to_datetime(road_closure_date, format="%d.%m.%Y")
    ].copy()
    return merged_data_after


def feature_target_selection(merged_data, road_closure_date, use_all_features, selected_features, targets, test_size, random_state):
    """
    Select features and targets from the dataset.
    """
    if use_all_features:
        selected_features_df = pd.read_csv("../train-data-all/variables.csv")
        selected_features = selected_features_df["Feature Name"].tolist()

    # Ensure 'date' column is not in the selected features list and it exists in the dataframe
    if 'date' in selected_features:
        selected_features.remove('date')
    assert 'date' in merged_data, "The dataframe does not contain a 'date' column"

    # Convert 'date' to datetime
    merged_data["date"] = pd.to_datetime(merged_data["date"], format="%d.%m.%Y")
    road_closure_datetime = pd.to_datetime(road_closure_date, format="%d.%m.%Y")

    # Separate the 'date' column for filtering
    dates = merged_data["date"]

    # Split data into features and targets
    X = merged_data[selected_features]
    y = merged_data[targets]

    # Ask the user which strategy they want to use
    train_after_closure = input("Do you want to include data after the road closure for training? \n "
                                "This means using test_size (yes/no): ").strip().lower() == "no"

    if train_after_closure:
        # Strategy 2: Train only with data before road closure, test with data after
        train_filter = dates < road_closure_datetime
        test_filter = ~train_filter  # Negation of train_filter

        X_train = X[train_filter]
        y_train = y[train_filter]
        X_test = X[test_filter]
        y_test = y[test_filter]
        X_test_dates = dates[test_filter]
    else:
        # Strategy 1: Train with 80% of data, test with 20% after road closure
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

        # Filter the temporary test set for after the road closure date
        temp_dates = dates.iloc[X_temp.index]
        test_filter = temp_dates > road_closure_datetime
        X_test = X_temp.loc[test_filter]
        y_test = y_temp.loc[test_filter]
        X_test_dates = temp_dates[test_filter]

    return X_train, X_test, y_train, y_test, X_test_dates





# Modify the train_model function to accept hyperparameters
def train_model(X_train, y_train, best_params=None):
    """
    Train a model using the training data.
    best_params are built upon different testing.
    """
    if not best_params:
        best_params = {
            'n_estimators': 1120,
            'random_state': 42,
            'max_depth': None,
            'bootstrap': True,
            'min_samples_leaf': 5,
            'min_samples_split': 12
        }

    # Initialize the imputer
    imputer = SimpleImputer(strategy='mean')

    # Impute missing values for X_train if necessary
    X_train_imputed = imputer.fit_transform(X_train)

    # Filter out NaN values from y_train and corresponding entries from X_train
    y_train_filtered = y_train.dropna()
    X_train_filtered = pd.DataFrame(X_train_imputed, index=X_train.index).loc[y_train_filtered.index]

    # Initialize the model with the best parameters
    model = RandomForestRegressor(**best_params)

    # Wrap the model with MultiOutputRegressor
    multioutput_model = MultiOutputRegressor(model)

    # Fit the model using the filtered (or imputed) training data
    multioutput_model.fit(X_train_filtered, y_train_filtered)

    return multioutput_model, imputer



def predict(model, imputer, X_test):
    """
    Make predictions using the trained model.
    """
    X_test_imputed = imputer.transform(X_test)
    y_pred = model.predict(X_test_imputed)
    return y_pred


def feature_importance_analysis(model, X):
    """
    Analyze and visualize feature importances.
    """
    feature_importances = model.estimators_[0].feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    top_features = X.columns[sorted_indices][:20]
    print("The top 20 features are:", top_features)
    tum_blue = '#3070B3'

    # Visualize the feature importances of only top 20 features
    plt.figure(figsize=(12, 6))
    plt.bar(top_features, feature_importances[sorted_indices][:20], orientation='vertical')
    plt.xticks(rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.grid(axis='y', color=tum_blue, linestyle='solid')
    plt.title('Top 20 feature importances for Südliche Au')
    plt.tight_layout()  # Adjusts the layout to prevent overlap
    plt.show(block=True)


# Before calling this function, ensure X_test_dates_filtered is properly filtered to match y_test_filtered and y_pred_filtered
def save_predictions_to_csv(X_test_dates, X_test, y_test, y_pred, filename="predictions/predictions_au.csv"):
    # Assuming X_test_dates is a Series with the same index as X_test, y_test, and y_pred
    X_test = X_test.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    # Combine the predicted values and the features into a results dataframe
    results_df = pd.concat([X_test, y_pred], axis=1)

    # Optionally, if you want to add the true values and predicted values side by side for comparison:
    for i, column in enumerate(y_test.columns):
        results_df[f"True_{column}"] = y_test[column]
        results_df[f"Predicted_{column}"] = y_pred.iloc[:, i]

    # Add the date column if necessary
    results_df.insert(0, 'date', X_test_dates.reset_index(drop=True))

    results_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")


def evaluate_model(y_test, y_pred, X_test, X_test_dates):
    # Convert y_pred to DataFrame for easier manipulation
    y_pred_df = pd.DataFrame(y_pred, index=y_test.index)

    # Filter out NaN values from y_test and corresponding entries in y_pred and X_test_dates
    mask = ~y_test.isnull().any(axis=1)
    y_test_filtered = y_test[mask]
    y_pred_filtered = y_pred_df[mask]
    X_test_dates_filtered = X_test_dates[mask]

    # Calculate metrics for each target variable
    for i, target in enumerate(y_test_filtered.columns):
        # Get the true and predicted values for the current target
        target_true = y_test_filtered[target]
        target_pred = y_pred_filtered.iloc[:, i]

        # Calculate metrics
        mae = mean_absolute_error(target_true, target_pred)
        r2 = r2_score(target_true, target_pred)

        # Output metrics for the current target
        print(f"Metrics for Target '{target}':")
        print("Mean Absolute Error:", mae)
        print("R-squared:", r2)

    # Calculate overall metrics for the complete model
    overall_mae = mean_absolute_error(y_test_filtered, y_pred_filtered)
    overall_r2 = r2_score(y_test_filtered, y_pred_filtered)

    # Output overall metrics
    print("\nMetrics for complete model:")
    print("Overall Mean Absolute Error:", overall_mae)
    print("Overall R-squared:", overall_r2)

    # # Plotting the scatterplot for the mean actual vs mean predicted values
    # plt.figure(figsize=(8, 5))
    # mean_true = y_test_filtered.mean(axis=1)
    # mean_pred = y_pred_filtered.mean(axis=1)
    # plt.scatter(mean_true, mean_pred, alpha=0.5)
    # plt.plot([mean_true.min(), mean_true.max()], [mean_true.min(), mean_true.max()], '--k')
    # plt.xlabel('Mean True Values')
    # plt.ylabel('Mean Predictions')
    # plt.title('Scatter plot for mean true vs. mean predicted for Südliche Au')
    # plt.grid(True)
    # plt.show(block=True)

    # Save the predictions to CSV
    save_predictions_to_csv(X_test_dates_filtered, X_test.loc[mask], y_test_filtered, y_pred_filtered)


def plot_predictions(y_test, y_pred, predictions_csv_path):
    predictions = pd.read_csv(predictions_csv_path)
    predictions['date'] = pd.to_datetime(predictions['date'])
    predictions.sort_values('date', inplace=True)

    true_columns = [col for col in predictions.columns if col.startswith('True_')]
    predicted_columns = [col for col in predictions.columns if col.startswith('Predicted_')]

    predictions['Mean_True'] = predictions[true_columns].mean(axis=1)
    predictions['Mean_Predicted'] = predictions[predicted_columns].mean(axis=1)

    tum_blue = '#072140'
    tum_lighter_blue = '#5E94D4'

    # Set up the matplotlib figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # First subplot: Scatter plot of mean true vs. mean predicted
    mean_true = y_test.mean(axis=1)
    mean_pred = pd.DataFrame(y_pred, index=y_test.index).mean(axis=1)

    ax1.scatter(mean_true, mean_pred, alpha=0.5)
    ax1.plot([mean_true.min(), mean_true.max()], [mean_true.min(), mean_true.max()], '--k')
    ax1.set_xlabel('Mean True Values')
    ax1.set_ylabel('Mean Predictions')
    ax1.set_title('Scatter Plot for Mean True vs. Mean Predicted')
    ax1.grid(True)

    # Second subplot: Time series of mean true and mean predicted
    ax2.plot(predictions['date'], predictions['Mean_True'], color=tum_blue, label='True mean relative speed',
             marker='o')
    ax2.plot(predictions['date'], predictions['Mean_Predicted'], color=tum_lighter_blue,
             label='Predicted mean relative speed', marker='x')

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.set_title('True and Predicted Mean Relative Speed Values Over Time')
    ax2.grid(True)
    ax2.legend()

    # Improve spacing between subplots
    plt.tight_layout()

    # Show the figure
    plt.show(block= True)


def main():
    config = get_config()

    datasets = load_data(config['data_files'], config['base_path'])
    merged_data = merge_data(datasets)
    features_to_process = ["84", "85", "86", "87", "88", "89", "90", "91", "92",
                               "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                               "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                               "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
                               "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
                               "41", "42", "43", "44", "45", "46", "47", "48", "49", "50",
                               "51", "52", "53", "54", "55", "56", "57", "58", "59", "60",
                               "61", "62", "63", "64", "65", "66", "67", "68", "69", "70",
                               "71", "72", "73", "74", "75", "76", "77", "78", "79", "80",
                               "81", "82", "83", "1_emoped", "2_emoped", "4_emoped", "5_emoped", "8_emoped",
                               "15_emoped", "87_emoped", "88_emoped",
                               "89_emoped", "90_emoped", "92_emoped", "21_actMode", "31_actMode", "41_actMode",
                               "3_actMode", "6_actMode", "7_actMode", "12_actMode",
                               "13_actMode", "14_actMode", "22_actMode", "23_actMode", "24_actMode", "32_actMode",
                               "33_actMode", "34_actMode", "42_actMode", "43_actMode",
                               "44_actMode", "52_actMode", "53_actMode", "54_actMode", "62_actMode", "63_actMode",
                               "1_actMode", "2_actMode", "4_actMode", "5_actMode",
                               "8_actMode", "15_actMode", "88_actMode", "45_actMode", "46_actMode", "87_actMode",
                               "89_actMode", "90_actMode", "92_actMode", "55_actMode",
                               "56_actMode", "64_actMode", "65_actMode", "72_actMode", "73_actMode", "9_actMode",
                               "10_actMode", "11_actMode", "16_actMode", "17_actMode",
                               "18_actMode", "25_actMode", "26_actMode", "27_actMode", "35_actMode", "36_actMode",
                               "37_actMode", "47_actMode", "48_actMode", "86_actMode",
                               "57_actMode", "58_actMode", "66_actMode", "67_actMode", "68_actMode", "91_actMode",
                               "74_actMode", "75_actMode", "76_actMode", "80_actMode",
                               "81_actMode", "19_actMode", "20_actMode", "27_actMode", "28_actMode", "29_actMode",
                               "38_actMode", "39_actMode", "49_actMode", "50_actMode",
                               "59_actMode", "60_actMode", "69_actMode", "70_actMode", "76_actMode", "77_actMode",
                               "78_actMode", "82_actMode", "83_actMode", "30_actMode",
                               "40_actMode", "51_actMode", "61_actMode", "71_actMode", "79_actMode", "1_escooter",
                               "2_escooter", "4_escooter", "5_escooter", "8_escooter", "15_escooter",
                               "87_escooter", "88_escooter", "89_escooter", "90_escooter", "92_escooter", "tavg",
                               "tmin", "tmax", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"] # Replace with your actual feature names
    road_closure_date = config['road_closure_date']
    merged_data = add_pre_closure_means_by_weekday(merged_data, features_to_process, road_closure_date)


    # Modify user-defined configurations based on user input
    config["grid_search"] = input(
        "Do you want to perform grid search tuning to fine-tune the hyperparameters? \n This "
        "will take up to 5 min (see parameter config to adjust) (yes/no): ").strip().lower() == "yes"
    config["use_all_features"] = input("Do you want to use all features from variables.csv instead of selected "
                                       "features? \n all features take a little longer (yes/no): ").strip().lower() == "yes"

    X_train, X_test, y_train, y_test, X_test_dates = feature_target_selection(
        merged_data,
        config['road_closure_date'],
        config["use_all_features"],
        config["selected_features"],
        config['targets'],
        config['test_size'],  # pass test_size here
        config['random_state']  # pass random_state here
    )
    if y_train.isnull().values.any():
        # Option 1: Drop rows with NaNs
        mask = ~y_train.isnull().any(axis=1)
        X_train = X_train[mask]
        y_train = y_train[mask]

    if config["grid_search"]:
        import grid_search_tuning
        best_params = grid_search_tuning.tune_hyperparameters(X_train, y_train)
        model, imputer = train_model(X_train, y_train, best_params)
    else:
        model, imputer = train_model(X_train, y_train)

    y_pred = predict(model, imputer, X_test)

    feature_importance_analysis(model, X_train)
    evaluate_model(y_test, y_pred, X_test, X_test_dates)  # Add X_test_dates as an argument
    plot_predictions(y_test, y_pred, '../predictions/predictions_au.csv')

if __name__ == "__main__":
    main()

