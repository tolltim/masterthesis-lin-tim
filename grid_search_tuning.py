from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


def tune_hyperparameters(X_train, y_train):
    """
    Tune hyperparameters of a RandomForestRegressor using GridSearchCV.
    """
    # Impute missing values in X_train
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)

    param_grid = {
        'n_estimators': [1120,1140,1160, 1180],
        'max_depth': [None,14,16],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    # Create a base model
    rf = RandomForestRegressor(random_state=42)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

    # Fit the grid search to the data
    grid_search.fit(X_train_imputed, y_train)

    print("Best Parameters Found: ", grid_search.best_params_)
    return grid_search.best_params_
