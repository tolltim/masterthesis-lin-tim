import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt

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
        'base_path': "train-data/",
        "grid_search": False,
        'road_closure_date': "12.06.2023",
        "use_all_features": False,
        "selected_features": ['87', '88', '88_escooter', '8_escooter', 'tsun', '2_escooter', 'tmax'],
        'targets': ['87', '88', '89', '90', '15', '34', '24', '92', '45'],  # these targets are the inner porject area of the süedliche au
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
        selected_features_df = pd.read_csv("train-data/variables.csv")
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
                                "This means test.size = 0.2 (yes/no): ").strip().lower() == "no"

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
            'min_samples_leaf': 1,
            'min_samples_split': 5
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
    plt.title('Top 20 feature importances for Südliche Au without 87 and 88 (training with data before road closure)')
    plt.tight_layout()  # Adjusts the layout to prevent overlap
    plt.show(block=True)


# Before calling this function, ensure X_test_dates_filtered is properly filtered to match y_test_filtered and y_pred_filtered
def save_predictions_to_csv(X_test_dates, X_test, y_test, y_pred, filename="predictions.csv"):
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
    print("\nMetrics for complete model")
    print("Overall Mean Absolute Error:", overall_mae)
    print("Overall R-squared:", overall_r2)

    # Save the predictions to CSV
    save_predictions_to_csv(X_test_dates_filtered, X_test.loc[mask], y_test_filtered, y_pred_filtered)


def main():
    config = get_config()

    road_closure_date = config['road_closure_date']
    datasets = load_data(config['data_files'], config['base_path'])
    merged_data = merge_data(datasets)

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

    if config["grid_search"]:
        import grid_search_tuning
        best_params = grid_search_tuning.tune_hyperparameters(X_train, y_train)
        model, imputer = train_model(X_train, y_train, best_params)
    else:
        model, imputer = train_model(X_train, y_train)

    y_pred = predict(model, imputer, X_test)

    feature_importance_analysis(model, X_train)
    evaluate_model(y_test, y_pred, X_test, X_test_dates)  # Add X_test_dates as an argument


if __name__ == "__main__":
    main()

