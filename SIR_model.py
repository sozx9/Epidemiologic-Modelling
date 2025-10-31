import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares

def sir_model(y0, t, beta, gamma):
    #S, I, R represents proportion of population that is susceptible, infected, and recovered respectively
    #S0, I0, R0 represent the initial proportions
    #rate of change of S, I, R can be found using known formulas
    #integrate the derivatives to get S, I, R across some time interval t (measured in terms of weeks)
    #output: values of S, I, R across time interval t

    def derivative(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    sol = odeint(derivative, y0, t, args=(beta, gamma))
    return sol.T  # returns S, I, R


def fit_sir_model(observed_cases, S0, I0, R0):
    #parameters: beta - represents transmission rate (between infected and susceptible individuals)
                #gamma - represents recovery rate
                #rho - represents reporting probability,
                      #used to map no. of true infections --> no. of reported infection cases
    
    #function estimates the above parameters using least squared error 
    #between observed counts (actual data) and SIR model counts, since the true parameters are unknown
    
    T = len(observed_cases)

    def residuals(params):
        beta, gamma, rho = params
        sol = sir_model(beta, gamma, S0, I0, R0, T)
        pred = simulate_observations(sol, rho)
        return pred - observed_cases
    
    init_guess = [0.5, 0.2, 0.8]
    bounds = ([0.01, 0.01, 0.1], [1, 1, 1])
    res = least_squares(residuals, init_guess, bounds=bounds)
    return res.x    


def simulate_observations(S, I, R, rho, N):
    #function generates synthetic data (i.e. no. of "observed" infection cases) for bootstrap forecasting later
    #we model the observed / reported number of cases as Poisson random variables,
    #with mean proportional to the no. of true new infections predicted by the SIR model

    dR = np.diff(R) 
    mean_cases = np.maximum(rho * dR * N, 1e-8)
    observed = np.random.poisson(mean_cases)
    return observed


def forecast(theta, N, training_cases, num_weeks_ahead):
    #parameter theta = (beta, gamma, rho), which are already fitted to the SIR model using least squares
    #function forecasts no. of cases num_weeks_ahead

    beta, gamma, rho = theta
    T = len(training_cases)

    #recreate entire trajectory up to T
    I0_guess = 0.001
    y0 = (1 - I0_guess, I0_guess, 0)
    S, I, R = sir_model(y0, np.arange(T+1), beta, gamma)

    #starting from last state
    y0_future = (S[-1], I[-1], R[-1])
    t_future = np.arange(num_weeks_ahead+1)

    S_fut, I_fut, R_fut = sir_model(y0_future, t_future, beta, gamma)
    future_obs = simulate_observations(S_fut, I_fut, R_fut, rho, N)
    return future_obs


def bootstrap_forecast(observed_train, N, num_weeks_ahead=4, num_samples=200):
    #function quantifies uncertainty (from parameter estimation and observation noise),
    #for a single forecast period, using parametric bootstraping by:
        #1. fitting the observed data once to obtain beta, gamma
        #2. for each bootstramp sample
            #a. generate synthetic data using obtained beta, gamma
            #b. re-fit parameters beta, gamma to synthetic data using least squares
            #c. perform forecasting
    
    #output: dict containing forecasts and prediction intervals for each forecast

    theta_hat = fit_sir_model(observed_train, N)
    forecasts = []

    for sample in range(num_samples):
        #simulate synthetic dataset under fitted theta_hat
        beta, gamma, rho = theta_hat
        I0_guess = 0.001
        y0 = (1 - I0_guess, I0_guess, 0)
        S, I, R = sir_model(y0, np.arange(len(observed_train)+1), beta, gamma)
        synthetic_obs = simulate_observations(S, I, R, rho, N)

        #re-fit to synthetic data
        theta = fit_sir_model(synthetic_obs, N)

        #forecast n_weeks_ahead with theta_b
        future = forecast_sir(theta, N, observed_train, num_weeks_ahead)
        forecasts.append(future)

    forecasts = np.array(forecasts)  # (B, horizon)
    median = np.median(forecasts, axis=0)
    lower = np.percentile(forecasts, 2.5, axis=0)
    upper = np.percentile(forecasts, 97.5, axis=0)
    return {"forecasts": forecasts, "median": median, "lower": lower, "upper": upper}
