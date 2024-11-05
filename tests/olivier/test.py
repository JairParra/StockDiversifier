#################################### 
###           1. Imports         ###
####################################
from src.beta_vae import BetaVAE, to_scalar, create_data_loaders, objective, train_beta_vae
from src.data_fetching import fetch_stock_data, scrape_sp500_wikipedia

import pandas as pd

# Deep learning
import torch

# Deep learning utilities
from sklearn.preprocessing import StandardScaler
import optuna
#################################### 
###           2. Testing         ###
####################################
# Load the uploaded stock data to understand its structure
stock_data_path = 'stock_data_clean.csv'
stock_data = pd.read_csv(stock_data_path)
stock_data= stock_data.dropna()

# Model dimensions
input_dim = stock_data.shape[1]
latent_dim = 10  # You can adjust this based on desired complexity

# Normalize the stock data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(stock_data)

# Assuming `normalized_data_cleaned` is the prepared dataset
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, normalized_data), n_trials=50)

# Best hyperparameters
print("Best hyperparameters:", study.best_params)