import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from autots import AutoTS
from IPython.display import Image
import json
from .utils_helpers import calculate_percentage_within_ci, suppress_stdout, check_constant_prediction
from .constants import TEST_YEAR, MIN_TRAINING_SAMPLES, CONFIDENCE_INTERVAL, FORECASTED_YEARS


def fit_autots_model(df, model_list, data_length = MIN_TRAINING_SAMPLES, forecast_length = 1, ensemble = None ):
    """
    Fits an AutoTS model to the specified data.

    Parameters:
    -----------
    df : pd.DataFrame
        The data frame containing the time series data.
    data_length : int, optional
        The minimum number of samples required to run the AutoTS model. Default is 20.
    forecast_length(int or None): 
        The number of periods to forecast into the future. If None, it will be set based on test data length.
    
    model_list (list of str): List of models to be used to fit the autots model
    ensemble(str or None): 
        Defines whether to apply esemble methods or not to AUTOTS model options are ('simple', 'horizontal', 'vertical', 'stacked', 'all')
        Default None
  

    Returns:
    --------
    dict
        A dictionary containing the fitted AutoTS model and its type, or None if there is not enough data or an error occurs.
    """

    
    # Initialize AutoTS model with specified parameters
    model = AutoTS(
        forecast_length=forecast_length, # Number of periods to forecast into the future
        frequency='infer',            # Frequency of the time series data ('infer' tries to auto-detect)
        prediction_interval=CONFIDENCE_INTERVAL,      # Confidence level for prediction intervals (e.g., 95%)
        ensemble= ensemble,                # Type of ensemble method ('simple', 'horizontal', 'vertical', 'stacked', 'all')
        transformer_list="fast",  # "superfast",
        max_generations=4,             # Maximum number of generations for evolutionary optimization
        num_validations=2,             # Number of cross-validation folds
        validation_method='backwards', # Method for cross-validation ('backwards', 'rolling', 'expanding')
        model_list=model_list,         # List of models to consider ('fast', 'default', 'all', or a custom list)
        no_negatives=True,             # Whether to restrict forecasted values to be non-negative
        n_jobs='auto'                  # Number of jobs to run in parallel ('auto' uses all available cores)
    )

    try:
        data = df
    
        
        if not data.empty:
            # Ensure data length is sufficient
            if len(data) < data_length:
                return {'model': None, 'type': 'No Data'}
            
            # Check if the series is constant
            elif data['Value'].nunique() == 1:
               #print(f"Time series for {country} - {animal_type} is constant. No model will be fitted.")
                return {'model': None, 'type': 'constant'}
            else:
            
                # Fit the AutoTS model with suppressed outpu
                with suppress_stdout():
                    model = model.fit(
                        data,
                        date_col='Year_datetime',
                        value_col='Value')
                    
                
                return {'model': model, 'type': 'AutoTS'}

        else:
            return {'model': None, 'type': 'No Data'}

    except ValueError as e:
        return {'model': None, 'type': 'Error'}
    

def evaluate_AUTOTS_model(df, country, animal_type, model_list =  ['ARIMA','FBProphet','ETS'], test_start_year = TEST_YEAR, ensemble = None):
    """
    Evaluate the forecasting model for a specific country and animal type.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the time series data with columns 'Year', 'Area', 'Item', and 'Value'.
    country : str
        The country for which the model is to be evaluated.
    animal_type : str
        The type of animal for which the model is to be evaluated.
    model_list (list of str): 
        List of models to be used to fit the autots model
    test_start_year : int
        The year from which the test data starts. Data before this year will be used for training,
        and data from this year onwards will be used for testing.
    ensemble(str or None): 
        Defines whether to apply esemble methods or not to AUTOTS model options are ('simple', 'horizontal', 'vertical', 'stacked', 'all')
        Default None

    Returns:
    --------
    tuple
        A tuple containing:
        - model_type : str or None
            Type of the selected model ('AutoTS', 'constant', 'No Data', etc.), or None if no data is available.
        - selected_model : dict or None
            Information about the selected model, if applicable (e.g., model parameters), or None if model is None.
        - best_fitness_score : float or None
            The fitness score of the selected model, if applicable, or None if model is None.
        - constant_prediction : bool
            Indicates if the model made constant predictions.
        - actual_values : pandas.Series or None
            The actual values of the test set, or None if no test data is available.
        - test_years : pandas.Series or None
            The years corresponding to the actual values, or None if no test data is available.
        - predictions : numpy.ndarray or None
            The predicted values for the test set, or None if model is None.
        - conf_int : numpy.ndarray or None
            The confidence intervals for the predicted values, or None if model is None or constant predictions.
        - proportion_in_ci : float or None
            The proportion of actual values that fall within the confidence intervals of the predicted values,
            or None if model is None, constant predictions, or no confidence intervals.

    """

  
    # Filter the data for the specified country and animal type
    train_data = df[(df['Area'] == country) & (df['Item'] == animal_type) & (df['Year'] < test_start_year)].copy()
    test_data = df[(df['Area'] == country) & (df['Item'] == animal_type) & (df['Year'] >= test_start_year)].copy()
    forecast_length = len( test_data) 
    if test_data.empty:
        # No test data available
        model_type = 'No Data'
        selected_model = None
        constant_prediction = False
        test_data['Value'] = None
        test_data['Year'] = None
        predictions = None
        conf_int = None
        proportion_in_ci = None

        return model_type, selected_model, constant_prediction, test_data['Value'], \
               test_data['Year'], predictions, conf_int, proportion_in_ci

    # Fit the model on the training data
    model_info = fit_autots_model(train_data, model_list, forecast_length = forecast_length, ensemble = ensemble)
 
    # Retrieve information about the selected model
    model = model_info['model']
    model_type = model_info['type']


    if model is None:
        # Model could not be fitted or no suitable model found
        model_type = 'Unknown Model'
        selected_model = None
        constant_prediction = False
        test_data['Value'] = None
        test_data['Year'] = None
        predictions = None
        conf_int = None
        proportion_in_ci = None

        return model_type, selected_model, constant_prediction, test_data['Value'], \
               test_data['Year'], predictions, conf_int, proportion_in_ci

    #check type of model emsemble in order to be able to access the type of model in the correct way
    if not model.ensemble:
        best_selected_model_spects= model.best_model
        selected_model = best_selected_model_spects['Model'].values[0]
    else:
        # Access information about the selected model
        best_selected_model_spects= model.best_model['ModelParameters']

        # Function to extract model name
        def extract_model_name(json_str):
            try:
                json_data = json.loads(json_str)
                return json_data.get('model_name')
            except json.JSONDecodeError:
                return None
        
        # Apply the function to extract model names
        model_names = best_selected_model_spects.apply(extract_model_name)
        
        # Get the extracted model name (assuming there's only one entry)
        selected_model = model_names.iloc[0] if not model_names.empty else None


    if model_type == 'constant':
        # Constant prediction model
        predictions = np.full(len(test_data), train_data['Value'].mean())
        conf_int = None
        constant_prediction = False

                
    elif model_type == 'AutoTS':
        # AutoTS model prediction
        # Forecast for the length of your test data
        prediction =  model.predict()
        # Extract predictions and intervals
        predictions = prediction.forecast
        lower_forecast = prediction.lower_forecast
        upper_forecast = prediction.upper_forecast
        conf_int = np.hstack((lower_forecast[['Value']].values, upper_forecast[['Value']].values))
        # Check if all predicted values are the same and generate a flag
        constant_prediction = np.all(predictions['Value'] == predictions['Value'][0])
        if constant_prediction:
            conf_int = None

    else:
        # Unknown model type
        model_type = 'Unknown Model'
        selected_model = None
        constant_prediction = False
        predictions = None
        conf_int = None
 
    # Calculate the proportion of actual values within the confidence intervals
    if conf_int is not None:
        proportion_in_ci = calculate_percentage_within_ci(test_data['Value'], conf_int)
    else:
        proportion_in_ci = None


    return model_type, selected_model, constant_prediction, test_data['Value'], test_data['Year'], predictions, conf_int, proportion_in_ci


def forecast_auto_ts(df, country, animal_type, model_list =  ['ARIMA','FBProphet','ETS'], number_of_years = FORECASTED_YEARS, ensemble = None):
    """
    Evaluate the forecasting model for a specific country and animal type.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the time series data with columns 'Year', 'Area', 'Item', and 'Value'.
    country : str
        The country for which the model is to be evaluated.
    animal_type : str
        The type of animal for which the model is to be evaluated.
    model_list : list of str
        List of models to be used to fit the AutoTS model.
    number_of_years : int
        The number of years for which the forecast is to be made. Default is 5 years.
    ensemble : str or None
        Defines whether to apply ensemble methods or not to AutoTS model. Options are ('simple', 'horizontal', 'vertical', 'stacked', 'all').
        Default is None.

    Returns:
    --------
    tuple
        A tuple containing:
        - model_type : str
            Type of the selected model ('AutoTS', 'constant', 'No Data', etc.), or None if no data is available.
        - selected_model : str or None
            Information about the selected model, if applicable (e.g., model name), or None if model is None.
        - constant_prediction : bool
            Indicates if the model made constant predictions.
        - predictions : numpy.ndarray or None
            The predicted values for the forecast period, or None if model is None.
        - conf_int : numpy.ndarray or None
            The confidence intervals for the predicted values, or None if model is None or constant predictions.
    """

  
    # Filter the data for the specified country and animal type
    data = df[(df['Area'] == country) & (df['Item'] == animal_type)]

    forecast_length = number_of_years
    print("FORECAST LENGTH")
    print(forecast_length)


    # Fit the model on the training data
    model_info = fit_autots_model(data, model_list, forecast_length = forecast_length, ensemble = ensemble)
 
    # Retrieve information about the selected model
    model = model_info['model']
    model_type = model_info['type']


    if model is None:
        # Model could not be fitted or no suitable model found
        model_type = 'Unknown Model'
        selected_model = None
        constant_prediction = False
        predictions = None
        conf_int = None


        return model_type, selected_model, constant_prediction, predictions, conf_int, 

    #check type of model emsemble in order to be able to access the type of model in the correct way
    if not model.ensemble:
        best_selected_model_spects= model.best_model
        selected_model = best_selected_model_spects['Model'].values[0]
    else:
        # Access information about the selected model
        best_selected_model_spects= model.best_model['ModelParameters']

        # Function to extract model name
        def extract_model_name(json_str):
            try:
                json_data = json.loads(json_str)
                return json_data.get('model_name')
            except json.JSONDecodeError:
                return None
        
        # Apply the function to extract model names
        model_names = best_selected_model_spects.apply(extract_model_name)
        
        # Get the extracted model name (assuming there's only one entry)
        selected_model = model_names.iloc[0] if not model_names.empty else None


    if model_type == 'constant':
        # Constant prediction model
        predictions = np.full(forecast_length, data['Value'].mean())
        conf_int = None
        constant_prediction = False

                
    elif model_type == 'AutoTS':
        # AutoTS model prediction
        # Forecast for the length of your test data
        prediction =  model.predict()
        # Extract predictions and intervals
        predictions = prediction.forecast
        lower_forecast = prediction.lower_forecast
        upper_forecast = prediction.upper_forecast
        conf_int = np.hstack((lower_forecast[['Value']].values, upper_forecast[['Value']].values))
        # Check if all predicted values are the same and generate a flag
        constant_prediction = np.all(predictions['Value'] == predictions['Value'][0])
        if constant_prediction:
            conf_int = None

    else:
        # Unknown model type
        model_type = 'Unknown Model'
        selected_model = None
        constant_prediction = False
        predictions = None
        conf_int = None
 



    return model_type, selected_model, constant_prediction, predictions, conf_int
    