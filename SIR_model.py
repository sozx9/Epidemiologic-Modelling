import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares

def run_sir_model(S0, I0, R0, num_weeks, transmission_rate, recovery_rate):
    #S, I, R represents proportion of population that is susceptible, infected, and recovered respectively
    #S0, I0, R0 represent the initial proportions
    #rate of change of S, I, R can be found using known formulas
    #integrate the derivatives to get S, I, R across some time interval t (measured in terms of weeks)
    #output: values of S, I, R across time interval t

    def get_derivative(S, I, R, transmission_rate, recovery_rate):
        #transmission rate = rate of transmission between S and I populations
        #recovery rate = rate of recovery in I populations
        
        dSdt = -transmission_rate * S * I
        dIdt = transmission_rate * S * I - recovery_rate * I
        dRdt = recovery_rate * I
        return dSdt, dIdt, dRdt

    weeks = np.arange(num_weeks)
    sol = odeint(get_derivative, [S0, I0, R0], weeks, args=(transmission_rate, recovery_rate))    
    return sol.T #returns S, I, R


def simulate_data(sol, reporting_rate):
    #function generates synthetic data (i.e. no. of "observed" infection cases) for bootstrap forecasting later
    #we model the observed / reported number of cases as Poisson random variables,
    #with mean proportional to the no. of true new infections predicted by the SIR model

    S, I, R = sol[:,0], sol[:,1], sol[:,2]
    mean = np.maximum(reporting_rate * I, 10**(-8))
    reported_cases = np.random.poisson(mean)
    return reported_cases


def fit_sir_params(data, S0, I0, R0):
    #function estimates parameters: transmission_rate, recovery_rate, reporting_rate 
    #using least squared error between I(t) and predicted I(t) from model

    I_actual = data
    num_weeks = len(data)
    
    def get_I_diff(params):
        #function to be used in least squares
        transmission_rate, recovery_rate, reporting_rate = params
        sol = run_sir_model(S0, I0, R0, num_weeks, transmission_rate, recovery_rate)
        I_predicted = simulate_data(sol, reporting_rate)
        return I_predicted - I_actual

    initial_guess = [0.5, 0.3, 0.8]
    bounds = ([0.01, 0.01, 0.1], [1, 1, 1])
    sol = least_squares(get_I_diff, initial_guess, bounds=bounds)
    return sol.x    


def run_forecast(params, last_training_state, num_weeks_ahead):
    #function predicts no. of infecton cases num_weeks_ahead, starting from some time
    #used to perform forecasting for a single bootstrap sample later
    
    transmission_rate, recovery_rate, reporting_rate = params
    S0, I0, R0 = last_training_state
    sol = run_sir_model(S0, I0, R0, num_weeks_ahead, transmission_rate, recovery_rate)    
    I = sol[:,1]
    forecast = rho * I
    return forecast 


def run_bootstrap(data, S0, I0, R0, num_weeks_ahead, num_samples):
    #runs parametric bootstrap, to see how the parameters vary, and how this affects the forecasted no. of infected cases
    #returns values of forecasted no. of cases needed to construct 95% prediction intervals
    
    fitted_params = fit_sir_params(data, S0, I0, R0)
    forecasts = []
    
    for _ in range(num_samples):
        #generate data from fitted model
        sol = run_sir_model(S0, I0, R0, len(data), fitted_params[0], fitted_params[1])
        generated_data = simulate_data(sol, fitted_params[2])

        #refit parameters to generated data
        refitted_params = fit_sir_params(generated_data, S0, I0, R0)

        #forecast with re-fitted parameters
        forecast = run_forecast(refitted_params, sol[-1], num_weeks_ahead)
        forecasts.append(forecast)
        
    forecasts = np.array(forecasts)
    median = np.median(forecasts, axis=0)
    lower = np.percentile(forecasts, 2.5, axis=0)
    upper = np.percentile(forecasts, 97.5, axis=0)
    return median, lower, upper


def rolling_origin_validation(data, window_size=30, forecast_horizon=4, num_bootstrap_samples=200):
    forecasts_stats = []

    for start in range(len(data) - window_size - forecast_horizon):
        training_data = data[start : start + window_size]
        testing_data = data[start + window_size : start + window_size + forecast_horizon]
        S0, I0, R0 = 0.99, 0.01, 0.0 

        mean, lower, upper = run_bootstrap(training_data, S0, I0, R0, forecast_horizon, num_bootstrap_samples)
        forecasts_stats.append({
            "test_start": start + window_size,
            "test_end": start + window_size + forecast_horizon,
            "median": median,
            "lower": lower,
            "upper": upper
        })
        
    return forecasts_stats
