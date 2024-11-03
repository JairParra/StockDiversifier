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
from torch.utils.data import DataLoader

# Deep learning utilities
import torchvision
from torchvision import datasets, transforms

# External tools and utilities
import kagglehub


## Configurations 

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

##########################
### 2. Utils Functions ###
##########################



###########################
### 3. Beta-VAE Objects ###
###########################

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean vector
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance vector

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_reconstructed = torch.sigmoid(self.fc_out(h))
        x_reconstructed = x_reconstructed.view(-1, 1, 28, 28)  # Reshape to original image size
        return x_reconstructed

class BetaVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, beta=1.0):
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
        # Compute the reconstruction loss (Binary Cross Entropy)
        recon_loss = F.binary_cross_entropy(x_reconstructed, x.view(-1, 1, 28, 28), reduction='sum')

        # Compute the KL divergence with the beta parameter
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss: recon_loss + beta * KL divergence
        return recon_loss + self.beta * kl_divergence


#########################
### 4. Model Training ###
#########################

# Function to train the model
def train_beta_vae(model, dataloader, optimizer, num_epochs=10):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        train_loss = 0
        for data, _ in dataloader:
            data = data.to(device)

            # Flatten the input to [batch_size, 784] for consistency with input_dim
            data = data.view(-1, 784)

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
        for data, _ in dataloader:
            data = data.to(device)  # Move data to the correct device
            data = data.view(-1, 784)  # Flatten the input for compatibility with the encoder

            # Forward pass
            recon_batch, mu, logvar = model(data)

            # Accumulate loss using the model's loss function
            test_loss += model.loss_function(recon_batch, data, mu, logvar).item()

    # Calculate average loss per sample
    test_loss /= len(dataloader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    # Visualize a few reconstructed images
    data, _ = next(iter(dataloader))  # Get a batch of test images
    data = data.to(device)
    data = data.view(-1, 784)  # Ensure input is flattened
    recon_batch, _, _ = model(data)  # Reconstruct the batch

    # Reshape for visualization
    data = data.view(-1, 1, 28, 28).cpu().detach()
    recon_batch = recon_batch.view(-1, 1, 28, 28).cpu().detach()

    # Display original and reconstructed images
    n = 8  # Number of images to display
    comparison = torch.cat([data[:n], recon_batch[:n]])  # Concatenate original and reconstructed images

    fig, axes = plt.subplots(2, n, figsize=(15, 3))
    for i in range(n):
        # Display original images in the top row
        axes[0, i].imshow(comparison[i][0], cmap="gray")
        axes[0, i].axis("off")

        # Display reconstructed images in the bottom row
        axes[1, i].imshow(comparison[i + n][0], cmap="gray")
        axes[1, i].axis("off")

    plt.show()
