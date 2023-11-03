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
        "use_all_features": False,
        "selected_features": ['58', 'removedparking-kol','weekday','87', 'tmax','89_emoped','tsun', '26', '90', '92',
                              '54', '55', '6', '70', 'bikedirection_north', 'pres', '88', '11'], #selecetd features based on importance and own understanding
        'targets': ['87', '24', '25', '34', '45', '44'], #these targets are the inner porject area of the süedliche au
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


def preprocess_data(merged_data, road_closure_date="12.06.2023"):
    """
    Preprocess data before and after the road closure date.
    """
    merged_data_after = merged_data[
        merged_data["date"] > road_closure_date].copy()  # Use copy to avoid SettingWithCopyWarning
    merged_data_before = merged_data[
        merged_data["date"] <= road_closure_date].copy()  # Use copy and corrected the condition

    merged_data_before["date"] = pd.to_datetime(merged_data_before["date"], format="%d.%m.%Y")
    merged_data_before["year"] = merged_data_before["date"].dt.year
    merged_data_before["month"] = merged_data_before["date"].dt.month
    merged_data_before["day"] = merged_data_before["date"].dt.day
    merged_data_before = merged_data_before.drop(["date"], axis=1)

    return merged_data_before, merged_data_after


def feature_target_selection(merged_data_before, merged_data_after, use_all_features, selected_features, targets):
    """
    Select features and targets from the dataset.
    """
    if use_all_features:
        selected_features_df = pd.read_csv("train-data/variables.csv")
        selected_features = selected_features_df["Feature Name"].tolist()

    y = merged_data_after[targets]
    X = merged_data_before[selected_features]

    # Make the longer dataset same lengths as the shorter one... maybe put that in methodology
    min_length = min(len(X), len(y))
    X = X.iloc[:min_length]
    y = y.iloc[:min_length]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Modify the train_model function to accept hyperparameters
def train_model(X_train, y_train, best_params=None):
    """
    Train a model using the training data.
    best param are built upon different testing
    """
    if not best_params:
        best_params = {
            'n_estimators': 1080,
            'random_state': 42,
            'max_depth': 14
        }

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)

    model = RandomForestRegressor(**best_params)
    multioutput_model = MultiOutputRegressor(model)
    multioutput_model.fit(X_train_imputed, y_train)

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

    # Visualize the feature importances of only top 20 features
    plt.figure(figsize=(12, 6))
    plt.bar(top_features, feature_importances[sorted_indices][:20], orientation='vertical')
    plt.xticks(rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.grid(axis='y', color='#D3D3D3', linestyle='solid')
    plt.title('Top 20 feature importances')
    plt.tight_layout()  # Adjusts the layout to prevent overlap
    plt.show(block=True)



def evaluate_model(y_test, y_pred):
    """
    Evaluate model performance.
    """
    for i, target in enumerate(y_test.columns):
        mae = mean_absolute_error(y_test[target], y_pred[:, i])
        r2 = r2_score(y_test[target], y_pred[:, i])
        print(f"Metrics for Target '{target}':")
        print("Mean Absolute Error:", mae)
        print("R-squared:", r2)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Metrics for complete model")
    print("mae: ", mae)
    print("r-squared: ", r2)


def main():
    config = get_config()

    datasets = load_data(config['data_files'], config['base_path'])
    merged_data = merge_data(datasets)
    merged_data_before, merged_data_after = preprocess_data(merged_data)

    # Modify user-defined configurations based on user input
    config["grid_search"] = input(
        "Do you want to perform grid search tuning to fine tune the hyperparameters? \n This "
        "will take a few minutes (yes/no): ").strip().lower() == "yes"
    config["use_all_features"] = input("Do you want to use all features from variables.csv instead of selected "
                                       "features? \n all features take a little (yes/no): ").strip().lower() == "yes"

    X_train, X_test, y_train, y_test = feature_target_selection(merged_data_before, merged_data_after,
                                                                config["use_all_features"], config["selected_features"],
                                                                config['targets'])

    if config["grid_search"]:
        import grid_search_tuning
        best_params = grid_search_tuning.tune_hyperparameters(X_train, y_train)
        model, imputer = train_model(X_train, y_train, best_params)
    else:
        model, imputer = train_model(X_train, y_train)

    y_pred = predict(model, imputer, X_test)

    feature_importance_analysis(model, X_train)
    evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    main()
