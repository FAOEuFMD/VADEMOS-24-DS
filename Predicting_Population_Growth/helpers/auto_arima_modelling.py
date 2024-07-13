import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from statsmodels.tsa.arima.model import ARIMAResults
from pmdarima.arima import AutoARIMA
from IPython.display import Image
from .utils_helpers import calculate_percentage_within_ci
from .constants import TEST_YEAR, MIN_TRAINING_SAMPLES, CONFIDENCE_INTERVAL, FORECASTED_YEARS



def fit_arima_model(df, country, animal_type, data_length = MIN_TRAINING_SAMPLES):
    """
    Fits an ARIMA or SARIMA model to the specified data.

    Parameters:
    df (pd.DataFrame): The data frame containing the time series data.
    country (str): The country for which the model is to be fitted.
    animal_type (str): The type of animal for which the model is to be fitted.
    data_length(int): The minimum number of samples to run the ARIMA model. Default is 20

    Returns:
    ARIMA: The fitted ARIMA model, or None if there is not enough data.
    """
    try:
        # Filter data for the specified animal and country
        data = df[(df['Area'] == country) & (df['Item'] == animal_type)].copy()
        
        if not data.empty:
            data['Year'] = pd.to_datetime(data['Year'], format='%Y')  # Convert 'Year' column to datetime format if needed
            # Convert year to index and grab only population value
            data = data.set_index('Year')['Value']
            # Set frequency explicitly as 'YE-DEC' (Year End, anchored to December)
            data.index = pd.DatetimeIndex(data.index.values, freq='infer')
            
            # Drop missing values
            data = data.dropna()

            # Ensure data length is sufficient
            if len(data) < data_length: 
                #print(f"Not enough data for {country} - {animal_type}")
                return {'model': None, 'type': 'No Data'}
            # Check if the series is constant
            elif data.nunique() == 1:
                print(f"Time series for {country} - {animal_type} is constant. Fitting ARMA(0,0,0) model.")
                # Fit ARIMA(0,0,0) model, what we simply do is to model a constant forecast
                return {'model': None, 'type': 'constant'}
            else: # Fit ARIMA  model using auto_arima
                arima_model = auto_arima(data, seasonal=False, suppress_warnings=True, error_action='ignore', stepwise=True)
                return {'model': arima_model, 'type': 'AutoARIMA'}

        else:
            #print(f"No data for {country} - {animal_type}")
            return {'model': None, 'type': 'No Data'}

    except ValueError as e:
        #print(f"Error fitting ARIMA model for {country} - {animal_type}: {e}")
        return {'model': None, 'type': 'Error'}


def evaluate_model(df, country, animal_type, test_start_year = TEST_YEAR, confidence_level = CONFIDENCE_INTERVAL):
    """
    Evaluate the ARIMA model for a specific country and animal type.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the time series data with columns 'Year', 'Area', 'Item', and 'Value'.
    country : str
        The country for which the model is to be evaluated.
    animal_type : str
        The type of animal for which the model is to be evaluated.
    test_start_year : int
        The year from which the test data starts. Data before this year will be used for training, and data from this
        year onwards will be used for testing.
    confidence_level:float
        The confidence level for the prediction intervals. Default at 0.95

    Returns:
    --------
    tuple
        A tuple containing:
        - mae : float
            Mean Absolute Error of the model's predictions on the test set.
        - mse : float
            Mean Squared Error of the model's predictions on the test set.
        - test : pandas.Series
            The actual values of the test set.
        - predictions : numpy.ndarray
            The predicted values for the test set.
        - proportion_in_ci : float
            The proportion of actual values that fall within the confidence_level intervals of the predicted values


    Example:
    --------
    >>> evaluate_model(df, 'Afghanistan', 'Asses', 2015)
    MAE for Afghanistan - Asses: 12345.67
    MSE for Afghanistan - Asses: 23456.78
    <matplotlib plot>

    This function ensures that the model is evaluated on separate training and testing sets to provide a proper
    assessment of its performance.
    """
    # Split the data into training and testing sets based on the test_start_year
    train_data = df[(df['Area'] == country) & (df['Item'] == animal_type) & (df['Year'] < test_start_year)].copy()
    # Filter the data for the specified country and animal type
    test_data = df[(df['Area'] == country) & (df['Item'] == animal_type) & (df['Year'] >= test_start_year)].copy()

    # Set index to datetime for test_data (if not already set)
    test_data.index = pd.to_datetime(test_data['Year'], format='%Y')
    
    if test_data.empty:
        #print(f"No test data available for {country} - {animal_type}")
        model_type = 'NoTestData'
        constant_prediction = False 
        mean_actual = None
        var_actual = None
        mae = None 
        mse = None
        test_data['Value'] = None
        test_data['Year'] = None 
        predictions = None 
        conf_int = None
        proportion_in_ci = None
        
        return model_type, constant_prediction, mean_actual, var_actual, mae, mse, test_data['Value'], test_data['Year'], predictions, conf_int, proportion_in_ci


    # Fit the model on the training data
    model_info  = fit_arima_model(train_data, country, animal_type)

    # Make predictions on the test data
    model = model_info['model']
    model_type = model_info['type']


    if model is None and model_type != 'constant':
        #print(f" {model_info['type']} for modeling {country} - {animal_type}")
        constant_prediction = False 
        mean_actual = None
        var_actual = None
        mae = None 
        mse = None
        test_data['Value'] = None
        test_data['Year'] = None 
        predictions = None 
        conf_int = None
        proportion_in_ci = None
        return model_type, constant_prediction, mean_actual, var_actual, mae, mse, test_data['Value'], test_data['Year'], predictions, conf_int, proportion_in_ci
    elif model_type == 'constant':
        # If ARIMA(0,0,0) model, predict constant value (mean of training data)
        predictions = np.full(len(test_data), train_data['Value'].mean())
        conf_int = None  # No confidence intervals for ARIMA(0,0,0)
    elif model_type == 'AutoARIMA':
        # If AutoARIMA model, predict using the fitted model
        predictions, conf_int = model.predict(n_periods=len(test_data), return_conf_int=True, alpha=1 - confidence_level)
    else:
        model_type = 'Unknown model'
        constant_prediction = False 
        mean_actual = None
        var_actual = None
        mae = None 
        mse = None
        test_data['Value'] = None
        test_data['Year'] = None 
        predictions = None 
        conf_int = None
        proportion_in_ci = None    
        return model_type, constant_prediction, mean_actual, var_actual, mae, mse, test_data['Value'], test_data['Year'], predictions, conf_int, proportion_in_ci

    #check if all predicted values are the same and generate a flag so that we can check those 
    constant_prediction = np.all(predictions == predictions[0])
        
    # Calculate the proportion of actual values within the confidence intervals
    if conf_int is not None and not constant_prediction:
        proportion_in_ci = calculate_percentage_within_ci(test_data['Value'], conf_int)
    else:
        proportion_in_ci = None  # Set proportion_in_ci to None if confidence 

    # Calculate error metrics
    mae = mean_absolute_error(test_data['Value'], predictions)
    mse = mean_squared_error(test_data['Value'], predictions)

    # report mean and var if in future we want to Standardize error metrics
    mean_actual = test_data['Value'].mean()
    var_actual = test_data['Value'].var()
    
    
    return model_type, constant_prediction, mean_actual, var_actual, mae, mse, test_data['Value'], test_data['Year'], predictions, conf_int, proportion_in_ci


def forecast_arima(df, country, animal_type, number_of_years = FORECASTED_YEARS, confidence_level = CONFIDENCE_INTERVAL):
    """
    Evaluate the ARIMA model for a specific country and animal type.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the time series data with columns 'Year', 'Area', 'Item', and 'Value'.
    country : str
        The country for which the model is to be evaluated.
    animal_type : str
        The type of animal for which the model is to be evaluated.
    number_of_years : int
        The number of years for which the forecast is to be made. Default is 5 years.
    confidence_level : float
        The confidence level for the prediction intervals. Default is 0.95.

    Returns:
    --------
    tuple
        A tuple containing:
        - model_type : str
            The type of model used.
        - constant_prediction : bool
            Flag indicating if the predictions are constant values.
        - predictions : numpy.ndarray
            The predicted values for the forecast period.
        - conf_int : numpy.ndarray or None
            The confidence intervals for the predictions, if available.
    """
    # Split the data into training and testing sets based on the test_start_year
    data = df[(df['Area'] == country) & (df['Item'] == animal_type)].copy()
  

    # Fit the model on the training data
    model_info  = fit_arima_model(data, country, animal_type)

    # Make predictions on the test data
    model = model_info['model']
    model_type = model_info['type']


    if model is None and model_type != 'constant':
        #print(f" {model_info['type']} for modeling {country} - {animal_type}")
        constant_prediction = False 
        predictions = None 
        conf_int = None
        return model_type, constant_prediction, predictions, conf_int
    
    elif model_type == 'constant':
        # If ARIMA(0,0,0) model, predict constant value (mean of training data)
        predictions = np.full(number_of_years, data['Value'].mean())
        conf_int = None  # No confidence intervals for ARIMA(0,0,0)
    elif model_type == 'AutoARIMA':
        # If AutoARIMA model, predict using the fitted model
        predictions, conf_int = model.predict(n_periods=number_of_years, return_conf_int=True, alpha=1 - confidence_level)
    else:
        model_type = 'Unknown model'
        constant_prediction = False 
        predictions = None 
        conf_int = None
  
        return model_type, constant_prediction,  predictions, conf_int

    #check if all predicted values are the same and generate a flag so that we can check those 
    constant_prediction = np.all(predictions == predictions[0])
           
        
    
    return model_type, model, constant_prediction,  predictions, conf_int