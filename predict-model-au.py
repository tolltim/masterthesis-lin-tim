
import joblib
from sklearn.model_selection import train_test_split

from rf_model_au import load_data, merge_data, get_config, evaluate_model, plot_predictions, \
    add_pre_closure_means_by_weekday, feature_importance_analysis
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
    features_to_process = ['wspd',  'outer_age', 'bikedirection_south',
                              'bikedirection_north', 'tavg',
                              'snow', 'pres', 'wpgt',
                               'outer_actmode',
                                 'prcp',   'tmax',
                               'inner_actmode',  'tmin', 'tsun',  'wdir',
                               'outer_speed',
                                'biketotal', 'inner_speed1','inner_speed2',
                              'inner_speed3','inner_speed4','inner_speed5','inner_speed6','inner_speed7','inner_speed8',
                           'inner_escooter', 'inner_emoped', 'inner_speed']
    road_closure_date = pd.to_datetime(config['road_closure_date'], format='%d.%m.%Y')
    merged_data = add_pre_closure_means_by_weekday(merged_data, features_to_process, road_closure_date)
    ###merged_data = merged_data[merged_data['date'] < '22.08.2023']## to ensure that they have the same length
    merged_data['date'] = pd.to_datetime(merged_data['date'], format='%d.%m.%Y')

    # Add the filter here

    #
    # These should match the features the wp_model was trained on
    selected_features =['wspd_', 'inner_age',  'outer_age' ,'newmobility','percentageclosedstreet',
                               'tavg_',
                               'bicycle-outer', 'removedparking', 'wpgt_',
                              'outer_actmode_',
                              'education-inner', 'prcp_',
                               'inner_actmode_', 'cultural-inner', 'tmin_',  'weekday', 'wdir_',
                              'transportation-inner', 'outer_speed_',
                               'biketotal_', 'inner_speed1_', 'inner_speed2_',
                              'inner_speed3_', 'inner_speed4_', 'inner_speed5_', 'inner_speed6_', 'inner_speed7_',
                              'inner_speed8_', 'inner_speed_'] ### need to be the same features as in wp model!!
    target_variables = ['inner_speed']### need to be the same target as in wp model!!
    merged_data = merged_data[merged_data['date'] > '2023-06-12']
    post_closure_data = merged_data[target_variables]
    pre_closure_data = merged_data[selected_features]
    selected_features_with_date = selected_features + ['date']

    # Modify your train-test split to include the date
    X_train, X_temp, y_train, y_temp = train_test_split(
        merged_data[selected_features_with_date], post_closure_data,
        test_size= 0.2, random_state=config['random_state'], shuffle=True
    )

    # Extract dates for X_test_dates
    # Extract dates for X_test_dates
    X_test_dates = X_temp['date']

    # Drop the date column before imputation and prediction
    X_temp = X_temp.drop(columns=['date'])
    X_train = X_train.drop(columns =['date'])

    # Predict for post-closure data
    #X_pre_closure = pre_closure_data[selected_features]
    X_pre_closure_imputed = imputer.transform(X_train)

    # Convert the imputed data back to a DataFrame, which is necessary for .loc indexing
    # Ensure that the columns match the original DataFrame's columns
    X_pre_closure_imputed_df = pd.DataFrame(X_pre_closure_imputed, columns=selected_features, index=X_train.index)

    # Apply the imputer to the test data
    X_temp_imputed = imputer.transform(X_temp)

    # Convert the imputed data back to a DataFrame, which is necessary for prediction
    X_temp_imputed_df = pd.DataFrame(X_temp_imputed, columns=selected_features, index=X_temp.index)

    # Assuming X_temp is your test set
    y_pred_post_closure = wp_model.predict(X_temp_imputed_df)

    # Ensure that y_actual_post_closure is aligned and has the same length
    y_actual_post_closure = y_temp

    # Extract actual post-closure values for comparison
    y_actual_post_closure = y_actual_post_closure[target_variables]

    # Concatenate the date, actual values, and predicted values into one DataFrame
    predictions_and_actual = pd.concat([
        X_test_dates.reset_index(drop=True),
        y_actual_post_closure.reset_index(drop=True).add_prefix('True_'),
        pd.DataFrame(y_pred_post_closure, columns=[f'Predicted_{col}' for col in target_variables])
    ], axis=1)

    predictions_csv_path = 'predictions/suedliche_au_predictions_based_on_wp.csv'
    predictions_and_actual.to_csv(path_or_buf=predictions_csv_path, index=False)


    # Plotting and evaluation functions can remain the same if they are expecting a DataFrame
    plot_predictions(y_test=y_actual_post_closure, y_pred=y_pred_post_closure,
                     predictions_csv_path='predictions/suedliche_au_predictions_based_on_wp.csv')
    #feature_importance_analysis(wp_model, X_train)

    # This will create an error, has to be neglected due to the fact, it is a function from another model
    evaluate_model(y_test=y_actual_post_closure, y_pred=y_pred_post_closure,
                   X_test=X_pre_closure_imputed_df, X_test_dates=X_test_dates)




if __name__ == "__main__":
    main()