# Epidemiologic Modelling
### 1. Project Status
This project is still in progress. 
Visualisations of forecasts and error analysis are still underway.

### 2. Project Overview
This project models the spread of COVID-19 in Singapore using epidemiological models, with a focus on evaluating the forecasting capabilities of the classical SIR (Susceptible-Infected-Recovered) model.

The project aims to answer the following questions:
* Does the SIR model describe the spread of COVID-19 in Singapore reasonably well?
* How well can the model perform short-term forecasts?
* What are the main limitations of the SIR model? How can we improve them?

### 3. Methods and Workflow
#### A. Modelling approach
* Implements a deterministic SIR model to describe infection dynamics.
* Parameters are estimated using least-squares on observed weekly cases.
* Observational noise is modelled using a Poisson process to model reporting uncertainty.

#### B. Forecasting and Uncertainty Quantification
* Performs parametric bootstrapping to simulate variability in model parameters, as well as generate a distribution of forecasts.
* Constructs 95% prediction intervals for short-term forecasts.
* Evaluates the model's predictive accuracy using rolling-origin validation.

#### C. Evaluation metrics
* Forecast mean and 95% prediction interval for each forecast period.
* Coverage: Proportion of actual values that fall within the prediction interval.
* Root mean squared error.

### 5. Dataset
Ministry of Health. (2023). Number of COVID-19 infections by Epi-week (2024) [Dataset]. data.gov.sg. Retrieved October 31, 2025 from https://data.gov.sg/datasets/d_11e68bba3b3c76733475a72d09759eeb/view

### 6. Future Extensions
Future work includes:
* Implementing a machine learning model, and comparing the short-term forecasting performances for both the SIR model and the machine learning model.
* Incorporating spatial or mobility data for more realistic modelling of transmission of COVID-19.
