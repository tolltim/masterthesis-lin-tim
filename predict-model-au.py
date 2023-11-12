
import joblib
from rf_model_au import load_data, merge_data, get_config,  evaluate_model, plot_predictions, add_pre_closure_means_by_weekday
import pandas as pd


def main():
    config = get_config()
    wp_model_path = 'wp_model.joblib'
    wp_model = joblib.load(wp_model_path)

    imputer_path = 'wp_imputer.joblib'
    imputer = joblib.load(imputer_path)

    # Load and prepare data
    datasets = load_data(config['data_files'], config['base_path'])
    merged_data = merge_data(datasets)
    merged_data['date'] = pd.to_datetime(merged_data['date'], format='%d.%m.%Y')
    features_to_process = []
    road_closure_date = pd.to_datetime(config['road_closure_date'], format='%d.%m.%Y')
    merged_data = add_pre_closure_means_by_weekday(merged_data, features_to_process, road_closure_date)
    post_closure_data = merged_data[merged_data['date'] > road_closure_date]
    dates_post_closure = post_closure_data['date']

    # These should match the features the wp_model was trained on
    selected_features =['wspd', 'inner_age', 'bicycle-inner', 'outer_age', 'bikedirection_south',
                              'bikedirection_north', 'tavg',
                              'snow', 'pres', 'bicycle-outer', 'removedparking', 'wpgt',
                               'outer_actmode',
                               'education-inner',  'prcp',   'consumption-outer', 'tmax',
                              'consumption-inner', 'inner_actmode', 'cultural-inner', 'tmin', 'tsun', 'weekday', 'wdir',
                                'transportation-inner','outer_speed', 'cultural-outer',
                               'education-outer', 'biketotal', 'transportation-outer','inner_speed1','inner_speed2',
                              'inner_speed3','inner_speed4','inner_speed5','inner_speed6','inner_speed7','inner_speed8'] ### need to be the same features as in wp model!!
    target_variables = ['inner_speed1', 'inner_speed2', 'inner_speed3', 'inner_speed4', 'inner_speed5', 'inner_speed6',
                        'inner_speed7', 'inner_speed8'] ### need to be the same target as in wp model!!

    # Predict for post-closure data
    X_post_closure = post_closure_data[selected_features]
    X_post_closure_imputed = imputer.transform(X_post_closure)

    # Convert the imputed data back to a DataFrame, which is necessary for .loc indexing
    # Ensure that the columns match the original DataFrame's columns
    X_post_closure_imputed_df = pd.DataFrame(X_post_closure_imputed, columns=selected_features, index=X_post_closure.index)

    y_pred_post_closure = wp_model.predict(X_post_closure_imputed_df)

    # Extract actual post-closure values for comparison
    y_actual_post_closure = post_closure_data[target_variables]

    # Concatenate the date, actual values, and predicted values into one DataFrame
    predictions_and_actual = pd.concat([dates_post_closure.reset_index(drop=True),
                                        y_actual_post_closure.reset_index(drop=True),
                                        pd.DataFrame(y_pred_post_closure,
                                                     columns=[f'Predicted_{col}' for col in target_variables])], axis=1)
    predictions_csv_path = 'suedliche_au_predictions_based_on_wp.csv'
    predictions_and_actual.to_csv(predictions_csv_path, index=False)

    # Plotting and evaluation functions can remain the same if they are expecting a DataFrame
    plot_predictions(y_test=y_actual_post_closure, y_pred=y_pred_post_closure,
                     predictions_csv_path=predictions_csv_path)

    # Make sure to pass the DataFrame to the evaluate_model function, not the NumPy array
    evaluate_model(y_test=y_actual_post_closure, y_pred=y_pred_post_closure,
                   X_test=X_post_closure_imputed_df, X_test_dates=dates_post_closure)



if __name__ == "__main__":
    main()