"""
data_febeta_vae.py
    This script contains functions to scrape the SP500 list from Wikpedia, 
    fetch stock data from Yahoo Finance AP, and then save the data locally to data_clean. 

@author: Olivier Makuch
"""

################## 
### 1. Imports ###
##################

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Deep learning utilities
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

##########################
### 2. Utils Functions ###
##########################

def to_scalar(data):
    """Convert a PyTorch tensor to a scalar."""    
    # Normalize the stock data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

def create_data_loaders(data, test_size=0.15, val_size=0.15, batch_size=64, random_state=42):
    '''Function to split the data into train, validation, and test sets and create DataLoaders.'''
    
    # Split data into training and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # Further split training data into training and validation sets
    train_data, val_data = train_test_split(train_data, test_size=val_size / (1 - test_size), random_state=random_state)
    
    # Create DataLoaders
    tr_loader = DataLoader(TensorDataset(torch.tensor(train_data, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.tensor(val_data, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(TensorDataset(torch.tensor(test_data, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    
    return tr_loader, va_loader, te_loader

def train_beta_vae(model, train_loader, val_loader=None, num_epochs=50, learning_rate=1e-3, beta=1):
    model.beta = beta
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = model.loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")
        
        # Evaluate on validation set if provided
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0]
                    recon_x, mu, logvar = model(x)
                    loss = model.loss_function(recon_x, x, mu, logvar)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
            model.train()  # Switch back to training mode
    
    return avg_loss  # Return the final training loss (or optionally validation loss)

def objective(trial, data):
    latent_dim = trial.suggest_categorical('latent_dim', [5, 10, 20])
    beta = trial.suggest_float('beta', 1, 10)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # Create DataLoaders with the suggested batch_size
    train_loader, val_loader, _ = create_data_loaders(data, batch_size=batch_size)
    
    # Initialize the model
    model = BetaVAE(input_dim=data.shape[1], latent_dim=latent_dim, beta=beta)
    
    # Train the model using the train loader and validate with the val loader
    avg_loss = train_beta_vae(model, train_loader, val_loader, num_epochs=10, learning_rate=learning_rate, beta=beta)
    
    return avg_loss

###########################
### 3. Beta-VAE Objects ###
###########################
# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3_mu = nn.Linear(256, latent_dim)
        self.fc3_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc3_mu(x)
        logvar = self.fc3_logvar(x)
        return mu, logvar

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = torch.sigmoid(self.fc3(z))  # Output in range [0, 1]
        return z

# Define the BetaVAE
class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta=1):
        super(BetaVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.beta = beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kld_loss
    