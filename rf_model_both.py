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
    The selected features and target variables can be seen in the train-data-all folder
    Innerspeed variables are selected from left top to right bottom of the area
    The target features describe the inner project area of the südiche au.
    """
    return {
        'data_files': [
            "demographics_agegroup.csv",
            "amenities_all.csv",
            "demographics_area_all.csv",
            "traffic-data-byquarters_all.csv",
            "strava-data_all.csv",
            "weather_data_all.csv",
            "escooter-counts_all.csv",
            "emopeds-counts_all.csv",
            "bike-data.csv"
        ],
        'base_path': "train-data-all/",  # folder for südliche au data
        "grid_search": False,
        'road_closure_date': "12.06.2023",  # closure date of südliche au
        "use_all_features": False,
        "selected_features": ["removedparking-kol", 'weekday',
                              'wspd_', 'bikedirection_south_',
                              'bikedirection_north_', 'tavg_', 'inner_speed8_au_',
                              'pres_', 'wpgt_', 'inner_speed1_au_',
                              'outer_actmode_au_', 'inner_speed7_au_',
                              'inner_speed4_au_', 'prcp_', 'inner_speed3_au_', 'tmax_',
                              'inner_actmode_au_', 'inner_speed5_au_', 'inner_speed6_au_', 'tmin_',
                              'tsun_', 'wdir_',
                              'inner_speed2_au_', 'outer_speed_au_',
                              'biketotal_', "inner_escooter_au_", "outer_escooter_au_", "inner_emoped_au_", "outer_emoped_au_", 'outer_actmode_wp_', 'inner_speed7_wp_',
                              'inner_speed4_wp_',  'inner_speed3_wp_',
                              'inner_actmode_wp_', 'inner_speed5_wp_', 'inner_speed6_wp_',
                              'inner_speed2_wp_', 'outer_speed_wp_'],
            # [ 'removedparking',
            #                     'weekday',
            #                   'wspd_', 'bikedirection_south_',"education-kol-inner", "education-landl-inner", "education-kol-outer",
    #                                  "education-landl-outer",
    #                                  "consumption-kol-inner", "consumption-landl-inner", "consumption-kol-outer",
    #                                  "consumption-landl-outer",
    #                                  "transportation-kol-inner", "transportation-landl-inner", "transportation-kol-outer",
    #                                  "transportation-landl-outer",
    #                                  "cultural-kol-inner", "cultural-landl-inner", "cultural-kol-outer",
    #                                  "cultural-landl-outer",
    #                                  "bicycle-kol-inner", "bicycle-landl-inner", "bicycle-kol-outer", "bicycle-landl-outer"
            #                   'bikedirection_north_', 'tavg_', 'inner_speed8_',
            #                   'pres_', 'wpgt_', 'inner_speed1_',
            #                   'outer_actmode_', 'inner_speed7_',
            #                   'inner_speed4_', 'prcp_', 'inner_speed3_', 'tmax_',
            #                   'inner_actmode_', 'inner_speed5_', 'inner_speed6_', 'tmin_',
            #                   'tsun_', 'wdir_',
            #                   'inner_speed2_', 'outer_speed_',
            #                   'biketotal_', "inner_escooter_", "outer_escooter_", "inner_emoped_", "outer_emoped_"],
        # these are the same features only adding the before name to hinder the model to predict future variables
        # with the same feature values, so for example, inner_speed1 would be predicted by inner_speed1, which makes no sense, so i invent those variables _before, with only the average after the road closure data
        'targets': ['inner_speed1_au','inner_speed2_au', 'inner_speed3_au', 'inner_speed4_au', 'inner_speed5_au', 'inner_speed6_au',
                    'inner_speed7_au', 'inner_speed8_au'],  # these targets are the inner project area of the süedliche au
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


def merge_data(datasets):
    """
    Merge datasets based on date.
    """
    merged_data = datasets["traffic-data-byquarters_all"].copy()
    for key, df in datasets.items():
        if key != "traffic-data-byquarters_all":
            merged_data = pd.merge(merged_data, df, on="date", how="outer")
    return merged_data


def add_pre_closure_means_by_weekday(merged_data, features, road_closure_date_str):
    merged_data['date'] = pd.to_datetime(merged_data['date'], format='%d.%m.%Y')

    # Convert the road_closure_date from string to datetime
    road_closure_date = pd.to_datetime(road_closure_date_str, format='%d.%m.%Y')

    for feature in features:
        new_col_name = f"{feature}_before"
        merged_data[new_col_name] = None

    # Copy values before the road closure and calculate means based on 'weekday'
    for feature in features:
        # Copy the values before the closure date
        merged_data.loc[merged_data['date'] <= road_closure_date, f"{feature}_"] = merged_data.loc[
            merged_data['date'] <= road_closure_date, feature]

        # Calculate mean for each weekday before the road closure
        means = merged_data.loc[merged_data['date'] <= road_closure_date].groupby('weekday')[feature].mean()

        # Assign the mean values based on weekday for the dates after the road closure
        for i, row in merged_data.loc[merged_data['date'] > road_closure_date].iterrows():
            weekday = row['weekday']
            merged_data.at[i, f"{feature}_before"] = means[weekday]

    # Return the updated DataFrame
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


def feature_target_selection(merged_data, road_closure_date, use_all_features, selected_features, targets, test_size,
                             random_state):
    """
    Select features and targets from the dataset.
    """
    use_all_features: False
    if use_all_features:
        selected_features_df = pd.read_csv("train-data-all-all/variables.csv")
        selected_features = selected_features_df["Feature Name"].tolist()

    # Ensure 'date' column is not in the selected features
    if 'date' in selected_features:
        selected_features.remove('date')
    assert 'date' in merged_data, "The dataframe does not contain a 'date' column"

    merged_data["date"] = pd.to_datetime(merged_data["date"], format="%d.%m.%Y")
    road_closure_datetime = pd.to_datetime(road_closure_date, format="%d.%m.%Y")

    # Split data into features and targets
    dates = merged_data["date"]

    X = merged_data[selected_features]
    y = merged_data[targets]

    # Ask the user for input
    train_after_closure = input("Do you want to include data after the road closure for training? \n "
                                "This means using test_size (yes/no): ").strip().lower() == "no"

    if train_after_closure:
        # Strategy 2: Train only with data before road closure, test with data after
        train_filter = dates < road_closure_datetime
        test_filter = ~train_filter  # r

        X_train = X[train_filter]
        y_train = y[train_filter]
        X_test = X[test_filter]
        y_test = y[test_filter]
        X_test_dates = dates[test_filter]
    else:
        # Strategy 1: Train with 80% of data, test with 20% after road closure
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                            shuffle=True)

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
    if not best_params:  # these are used if no tuning is selected from the user input
        best_params = {
            'n_estimators': 1120,
            'random_state': 42,
            'max_depth': None,
            'bootstrap': True,
            'min_samples_leaf': 5,
            'min_samples_split': 2
        }


    imputer = SimpleImputer(strategy='mean')

    # Impute missing values for X_train if necessary
    X_train_imputed = imputer.fit_transform(X_train)

    # Filter out NaN values from y_train and corresponding entries from X_train
    y_train_filtered = y_train.dropna()
    X_train_filtered = pd.DataFrame(X_train_imputed, index=X_train.index).loc[y_train_filtered.index]

    # Initialize the model with the best parameters
    model = RandomForestRegressor(**best_params)

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

    results_df = pd.concat([X_test, y_pred], axis=1)

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
    plt.show(block=True)


def main():
    config = get_config()

    datasets = load_data(config['data_files'], config['base_path'])
    merged_data = merge_data(datasets)
    features_to_process = ['wspd', 'bikedirection_south',
                           'bikedirection_north', 'tavg', 'inner_speed8_au',
                           'snow', 'pres', 'wpgt', 'inner_speed1_au',
                           'outer_actmode_au', 'inner_speed7_au',
                           'inner_speed4_au', 'prcp', 'inner_speed3_au', 'tmax',
                           'inner_actmode_au', 'inner_speed5_au', 'inner_speed6_au', 'tmin', 'tsun', 'wdir',
                           'inner_speed2_au', 'outer_speed_au',
                           'biketotal', "inner_escooter_au", "outer_escooter_au", "inner_emoped_au", "outer_emoped_au", 'inner_speed8_wp',
                           'inner_speed1_wp',
                           'outer_actmode_wp', 'inner_speed7_wp',
                           'inner_speed4_wp',  'inner_speed3_wp',
                           'inner_actmode_wp', 'inner_speed5_wp', 'inner_speed6_wp',
                           'inner_speed2_wp', 'outer_speed_wp']  # Replace with your actual feature names
    road_closure_date = config['road_closure_date']
    merged_data = add_pre_closure_means_by_weekday(merged_data, features_to_process, road_closure_date)

    # Modify user-defined configurations based on user input
    config["grid_search"] = input(
        "Do you want to perform grid search tuning to fine-tune the hyperparameters? \n This "
        "will take up to 5 min (see parameter config to adjust) (yes/no): ").strip().lower() == "yes"
    #config["use_all_features"] = input("Do you want to use all features from variables.csv instead of selected "
     #                                  "features? \n all features take a little longer (yes/no): ").strip().lower() == "yes"

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
    # save_predictions_to_csv(X_test_dates_filtered, X_test.loc[mask], y_test_filtered, y_pred_filtered)
    plot_predictions(y_test, y_pred, 'predictions/predictions_au.csv')


if __name__ == "__main__":
    main()
