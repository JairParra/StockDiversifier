"""
data_febeta_vae.py
    This script contains functions to scrape the SP500 list from Wikpedia, 
    fetch stock data from Yahoo Finance AP, and then save the data locally to data_clean. 

@author: Olivier Makuch
"""

################## 
### 1. Imports ###
##################

# General
import os

# Data science 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Deep learning utilities
import torchvision
from torchvision import datasets, transforms

## Configurations 

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

##########################
### 2. Utils Functions ###
##########################

from data_fetching import fetch_stock_data
from data_fetching import scrape_sp500_wikipedia

sp500_df = scrape_sp500_wikipedia()  # Use the function you created to scrape S&P 500 companies
custom_tickers = ['TSLA', 'ZM', 'SNOW']  # Example custom tickers

stock_data, sector_mapping, industry_mapping = fetch_stock_data(sp500_df, custom_tickers)

###########################
### 3. Beta-VAE Objects ###
###########################
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, latent_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean vector
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance vector

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=None):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        if output_dim is None:
            raise ValueError("output_dim must be specified and match the input_dim of the Encoder.")
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_reconstructed = torch.sigmoid(self.fc_out(h))
        return x_reconstructed

class BetaVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, latent_dim=20, beta=1.0):
        super(BetaVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.beta = beta  # Beta parameter for the KL divergence term

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, x_reconstructed, x, mu, logvar):
        # Reconstruction loss (MSE or another suitable loss for tabular data)
        recon_loss = F.mse_loss(x_reconstructed, x, reduction='sum')

        # KL divergence with beta parameter
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss: recon_loss + beta * KL divergence
        return recon_loss + self.beta * kl_divergence

#########################
### 4. Model Training ###
#########################

def train_beta_vae(model, dataloader, optimizer, num_epochs=10):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        train_loss = 0
        for data in dataloader:
            data = data.to(device)
            
            # No need to flatten the data, as stock data should already be in the correct shape

            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, logvar = model(data)

            # Compute loss using the model's loss function
            loss = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()

            optimizer.step()

        # Calculate average loss per sample in the epoch
        train_loss /= len(dataloader.dataset)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

    return train_losses


#################################
### 5. Evaluation and Testing ###
#################################

def test_beta_vae(model, dataloader):
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for data in dataloader:
            data = data.to(device)  # Move data to the correct device

            # Forward pass
            recon_batch, mu, logvar = model(data)

            # Accumulate loss using the model's loss function
            test_loss += model.loss_function(recon_batch, data, mu, logvar).item()

    # Calculate average loss per sample
    test_loss /= len(dataloader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    return test_loss

#################################
### 6. Data Preperation       ###
#################################

stock_data_raw = stock_data.copy()

#### One hot encode the 'Sector' column ####
# Map the Sector names to their respective codes
stock_data['Sector'] = stock_data['Sector'].map(sector_mapping)

# One-hot encode the 'Sector_Code' column
one_hot_encoded_sectors = pd.get_dummies(stock_data['Sector'], prefix='Sector')

# Combine the one-hot encoded columns with the original DataFrame
stock_data = pd.concat([stock_data, one_hot_encoded_sectors], axis=1)

# Drop the 'Sector_Code' column if you don't need it
stock_data.drop('Sector', axis=1, inplace=True)

#### One hot encode the 'Industry' column ####
# Map the Industry names to their respective codes
stock_data['Industry'] = stock_data['Industry'].map(industry_mapping)

# One-hot encode the 'Industry_Code' column
one_hot_encoded = pd.get_dummies(stock_data['Industry'], prefix='Industry')

# Combine the one-hot encoded columns with the original DataFrame
stock_data = pd.concat([stock_data, one_hot_encoded], axis=1)

# Drop the 'Industry_Code' column if you don't need it
stock_data.drop('Industry', axis=1, inplace=True)

# Drop the Ticker and Company Name columns
stock_data.drop('Ticker', axis=1, inplace=True)
stock_data.drop('Company Name', axis=1, inplace=True)

print(stock_data.head())
print(stock_data.columns)
#################################
### 7. Transform Data         ###
#################################

print(stock_data.values)

stock_data.to_csv('stock_data_one_hot.csv', index=False)
# Assuming stock_data is a pandas DataFrame with 20 features
stock_tensor = torch.tensor(stock_data.values, dtype=torch.float32)


# Split the data into training and testing sets
train_size = int(0.8 * len(stock_tensor))
test_size = len(stock_tensor) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(stock_tensor, [train_size, test_size])

# Create data loaders
batch_size = 32  # You can adjust this
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the Beta-VAE model
# Number of features in the stock dataset
input_dim = 20
# .to(device)
beta_vae = BetaVAE(input_dim=input_dim)

# Define optimizer
optimizer = optim.Adam(beta_vae.parameters(), lr=1e-3)

# Train the Beta-VAE
train_losses = train_beta_vae(beta_vae, train_loader, optimizer, num_epochs=stock_data.shape[1])

# Test the VAE
test_beta_vae(beta_vae, test_loader)  # Use the correct model instance name