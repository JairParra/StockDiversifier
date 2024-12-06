"""
pca_enc.py
    This script defines a PCAEncoder class for dimensionality reduction and encoding.

@author: Hair Parra
"""

##################
### 1. Imports ###
##################

import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Custom modules
from src.data_fetching import prepare_data_for_vae

############################
### 2. Utility Functions ###
############################

def generate_embeddings_dict(stock_df, scaler, pca_encoder, ticker_col="Ticker"):
    """
    Process a stock DataFrame, normalize it, and generate PCA embeddings for each row identified by the "ticker" column.
    
    Parameters:
    - stock_df (pd.DataFrame): The input DataFrame containing stock data, including a "ticker" column.
    - scaler (sklearn.preprocessing.StandardScaler): A prefitted scaler for normalization.
    - pca_encoder (PCAEncoder): The pretrained PCA encoder to generate embeddings.
    - ticker_col (str): Column name in the DataFrame that holds the stock ticker symbols.
    
    Returns:
    - dict: A dictionary where keys are ticker symbols, and values are corresponding PCA embeddings.
    """
    # Step 1: Prepare the data for PCA preprocessing
    stock_data_pca = prepare_data_for_vae(stock_df)  # Assuming this handles missing values and feature selection.
    
    # Step 2: Normalize the data using the prefitted scaler
    norm_data_full = scaler.transform(stock_data_pca)
    
    # Step 3: Generate embeddings using the PCA encoder
    pca_embeddings = pca_encoder.transform(norm_data_full)
    
    # Step 4: Match embeddings to the original tickers
    tickers = stock_df[ticker_col].values  # Ensure alignment between tickers and rows in stock_data_pca
    embeddings_dict = {ticker: pca_embeddings[i] for i, ticker in enumerate(tickers)}
    
    return embeddings_dict


######################
### 3. PCA Encoder ###
######################

class PCAEncoder:
    def __init__(self, data: np.ndarray = None, n_components: int = None):
        """
        Initialize PCAEncoder with specified number of components.
        
        Parameters:
        - data: Input data (numpy array). Assumes data is already scaled.
        - n_components: Number of PCA components. If None, all components are used.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self._data = None  # Internal storage for the original data

        if data is not None:
            self.fit(data)

    def fit(self, data: np.ndarray):
        """
        Fit the PCA model to the data and store the data internally.
        
        Parameters:
        - data: Input data (numpy array). Assumes data is already scaled.
        """
        self._data = data  # Save the data for future use
        self.pca.fit(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using the fitted PCA model.
        
        Parameters:
        - data: Input data (numpy array). Assumes data is already scaled.
        
        Returns:
        - PCA-transformed data (numpy array).
        """
        return self.pca.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit PCA model and transform the data.
        
        Parameters:
        - data: Input data (numpy array). Assumes data is already scaled.
        
        Returns:
        - PCA-transformed data (numpy array).
        """
        self.fit(data)
        return self.transform(data)

    def save(self, path: str):
        """
        Save the PCAEncoder to a file.
        
        Parameters:
        - path: Path to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "PCAEncoder":
        """
        Load a PCAEncoder from a file.
        
        Parameters:
        - path: Path to the saved model.
        
        Returns:
        - Loaded PCAEncoder instance.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    @property
    def explained_variance_ratio_(self) -> float:
        """
        Get the cumulative explained variance ratio for the current PCA configuration.
        
        Returns:
        - A float representing the percentage of total variance explained by the current components.
        """
        if not hasattr(self.pca, "explained_variance_ratio_"):
            raise ValueError("PCA model must be fitted before accessing explained_variance_ratio_.")

        # Cumulative explained variance for the selected components
        current_cumulative_variance = np.sum(self.pca.explained_variance_ratio_)
        return current_cumulative_variance

    def autotune_num_components(self, explained_var: float = 0.95):
        """
        Automatically determine the minimum number of components to retain the desired explained variance.
        
        Parameters:
        - explained_var: Target explained variance (default is 0.95).
        
        Updates:
        - self.n_components: The number of components to achieve the desired explained variance.
        """
        if self._data is None:
            raise ValueError("No data available for autotuning. Fit the PCA model first or provide data during initialization.")
        
        # Use all components to determine the optimal number
        full_pca = PCA()
        full_pca.fit(self._data)
        cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
        optimal_components = np.searchsorted(cumulative_variance, explained_var) + 1

        # Update and refit the PCA model
        self.n_components = optimal_components
        self.pca = PCA(n_components=optimal_components)
        self.pca.fit(self._data)

    def reset_num_components(self, n_components: int = None):
        """
        Reset the number of components and refit the PCA model.
        
        Parameters:
        - n_components: New number of components. If None, all components are used.
        """
        if self._data is None:
            raise ValueError("No data available to reset. Fit the PCA model first or provide data during initialization.")
        
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self._data)
