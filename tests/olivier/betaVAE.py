# ==============================================================================
# title : betaVAE.py
# Author : Olivier Makuch
# Date   : 2024-11-20
# Purpose: Extract stock embeddings to test portfolio diversity
# ==============================================================================

# ==============================================================================
# Load libraries
# ==============================================================================

# Standard libraries
import pandas as pd 
import json
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import optuna
from sklearn.model_selection import train_test_split

# Custom Classes
from src.beta_vae import Encoder
from src.beta_vae import Decoder
from src.beta_vae import BetaVAE

# Custom Functions
from src.beta_vae import create_data_loaders
from src.beta_vae import objective
from src.beta_vae import train_beta_vae
from src.beta_vae import get_embeddings
from src.data_fetching import scrape_sp500_wikipedia
from src.data_fetching import fetch_stock_data
from src.data_fetching import prepare_data_for_vae

# ==============================================================================
# Fetch the data and prepare it for the VAE
# ==============================================================================

sp500_df = scrape_sp500_wikipedia()  # Use the function you created to scrape S&P 500 companies
custom_tickers = ['TSLA', 'ZM', 'SNOW']  # Example custom tickers
stock_data, sector_mapping, industry_mapping = fetch_stock_data(sp500_df, custom_tickers) # Fetch data
stock_data_vae = prepare_data_for_vae(stock_data)  # Prepare data for VAE

# ==============================================================================
# Pre-processing for training 
# ==============================================================================

stock_data= stock_data_vae.dropna()

# Normalize the stock data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(stock_data)

# Convert the normalized data to PyTorch tensors
tensor_data = torch.tensor(normalized_data, dtype=torch.float32)

train_loader, val_loader, test_loader = create_data_loaders(tensor_data)


# Check the DataLoader output
next(iter(train_loader))[0].shape  # Shape of one batch of data

# ==============================================================================
# Hyper parameter tuning 
# ==============================================================================

# Define the file path for storing the best parameters
best_params_file = "config/best_params.json"

# Check if the best_params file exists
if os.path.exists(best_params_file):
    # Load the best parameters from the file
    with open(best_params_file, "r") as f:
        best_params = json.load(f)
    print("Loaded best parameters from file:", best_params)
else:
    # File doesn't exist, run the Optuna study
    print("Best parameters file not found. Running Optuna study...")
    
    # Assuming `normalized_data_cleaned` is the prepared dataset
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, normalized_data), n_trials=50)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters found:", best_params)

    # Save the best parameters to a JSON file
    os.makedirs(os.path.dirname(best_params_file), exist_ok=True)  # Create directory if it doesn't exist
    with open(best_params_file, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Best parameters saved to {best_params_file}")

# ==============================================================================
# Load or train the model
# ==============================================================================

# File path for the saved model
model_path = "models/beta_vae_with_metadata.pth"

# Check if the model file exists
if os.path.exists(model_path):
    print(f"Model found at {model_path}. Loading the model...")

    # Load model state and metadata
    model_metadata = torch.load(model_path)

    # Recreate the model using the saved metadata
    beta_vae = BetaVAE(
        input_dim=model_metadata["input_dim"],
        latent_dim=model_metadata["latent_dim"],
        beta=model_metadata["beta"]
    )
    beta_vae.load_state_dict(model_metadata["model_state"])
    beta_vae.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")
else:
    print(f"Model not found at {model_path}. Training a new model...")

    # Assuming you use the best hyperparameters from Optuna
    latent_dim = best_params['latent_dim']
    beta = best_params['beta']
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    input_dim = stock_data.shape[1]

    # Recreate DataLoaders with the best batch size
    train_loader, val_loader, test_loader = create_data_loaders(tensor_data, batch_size=batch_size)

    # Initialize and train the model
    beta_vae = BetaVAE(input_dim=input_dim, latent_dim=latent_dim, beta=beta)
    train_beta_vae(beta_vae, train_loader, val_loader, num_epochs=50, learning_rate=learning_rate)

    # Save the trained model with metadata
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure the directory exists
    model_metadata = {
        "model_state": beta_vae.state_dict(),
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "beta": beta
    }
    torch.save(model_metadata, model_path)
    print(f"Model trained and saved to {model_path}.")

# ==============================================================================
# Extract embeddings for the stock data
# ==============================================================================

# Extract embeddings for the training data
train_embeddings = get_embeddings(beta_vae, train_loader)

# Optionally, extract embeddings for validation and test data
val_embeddings = get_embeddings(beta_vae, val_loader)
test_embeddings = get_embeddings(beta_vae, test_loader)

print("Train Embeddings Shape:", train_embeddings)  # Should be (num_samples, latent_dim)


