"""
portfolio.py
    This script contains a number of financial portfolio related functions and utils
    including return calculations, risk calculations, and portfolio optimization.

@author: Hair Parra
"""

################## 
### 1. Imports ###
##################

# General 
import warnings
import yfinance as yf
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

# Data Manipulation & Visualiztion  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optimization 
from scipy.optimize import minimize

##########################
### 2. Utils Functions ###
##########################

def fetch_and_calculate_returns(tickers: List[str], 
                                period: str = '1y', 
                                interval: str = '1wk', 
                                price_column: str = 'Close') -> Dict[str, np.ndarray]:
    """
    Fetch historical data for a list of tickers and calculate returns for each ticker.
    
    Parameters:
    - tickers: A list of stock ticker symbols to fetch data for.
    - period: The period for which to fetch data (default is '1y').
    - interval: The interval for the data (e.g., '1d', '1wk', '1mo').
    - price_column: The column name to use for price ('Open', 'Close', 'Adj Close', etc.).
    
    Returns:
    - returns_dict: A dictionary where keys are tickers and values are numpy arrays of historical returns.
    """
    returns_dict = {}
    
    for ticker in tqdm(tickers, desc="Fetching Data", unit="ticker"):
        try:
            # Fetch historical data
            stock_data = yf.Ticker(ticker).history(period=period, interval=interval)
            
            if stock_data.empty:
                print(f"No data found for ticker: {ticker}")
                continue
            
            # Ensure the specified price column exists
            if price_column not in stock_data.columns:
                raise ValueError(f"Price column '{price_column}' not found for ticker: {ticker}")
            
            # Calculate returns
            prices = stock_data[price_column].values
            returns = np.diff(prices) / prices[:-1]  # Simple returns
            returns_dict[ticker] = returns
        
        except Exception as e:
            print(f"Failed to fetch or process data for {ticker}: {str(e)}")
    
    return returns_dict


##########################
### 3. Portfolio Class ###
##########################

class Portfolio:
    def __init__(self, 
                 returns_dict: Dict[str, np.ndarray], 
                 frequency: str, 
                 weights: Optional[Dict[str, float]] = None, 
                 risk_free:float = 0.00) -> None:
        """
        Initializes the Portfolio class.
        
        Parameters:
        - returns_dict: A dictionary where keys are tickers and values are numpy arrays of returns.
        - frequency: The frequency of the returns. Must be one of "1min", "5min", "hourly", "daily", "weekly", "monthly".
        - weights: Optional; a dictionary with tickers as keys and portfolio weights as values.
        """
        valid_frequencies = {"1min", "5min", "hourly", "daily", "weekly", "monthly"}
        if frequency not in valid_frequencies:
            raise ValueError(f"Frequency must be one of {valid_frequencies}.")
        self.frequency = frequency

        # Save risk free rate 
        self.risk_free = risk_free

        # Store tickers and raw returns
        self.tickers = list(returns_dict.keys())
        self.raw_returns = returns_dict

        # Initialize weights
        if weights is None:
            weights = {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}
        else:
            if set(weights.keys()) != set(self.tickers):
                raise ValueError("Weights must contain the same tickers as returns_dict.")
            total_weight = sum(weights.values())
            if not np.isclose(total_weight, 1.0):
                warnings.warn("Weights do not sum to 1. Normalizing weights.")
                weights = {ticker: w / total_weight for ticker, w in weights.items()}
        self.w = np.array([weights[ticker] for ticker in self.tickers])

        # Calculate covariance matrix (Sigma) and its diagonal (sigmas)
        self.returns_matrix = np.column_stack([returns_dict[ticker] for ticker in self.tickers])
        self.Sigma = np.cov(self.returns_matrix, rowvar=False)
        self.sigmas = np.sqrt(np.diag(self.Sigma))

        # Portfolio expected return and volatility
        self.expected_returns = self.returns_matrix.mean(axis=0)  # Mean returns
        self.portfolio_expected_return = np.dot(self.w, self.expected_returns)
        self.portfolio_volatility = np.sqrt(np.dot(self.w.T, np.dot(self.Sigma, self.w)))

        # Calculate initial diversification ratio and sharpe ratio
        self.diversification_ratio = self._calculate_diversification_ratio()
        self.sharpe_ratio = self._calculate_sharpe_ratio(risk_free_rate=self.risk_free)

    def _calculate_diversification_ratio(self) -> float:
        """
        Calculate the diversification ratio of the portfolio.
        
        Returns:
        - The diversification ratio as a float.
        """
        weighted_sigma = np.dot(self.w, self.sigmas)
        portfolio_volatility = self.portfolio_volatility
        return weighted_sigma / portfolio_volatility
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float) -> float:
        """
        Calculate the Sharpe ratio of the portfolio.
        
        Parameters:
        - risk_free_rate: The risk-free rate of return.
        
        Returns:
        - The Sharpe ratio as a float.
        """
        excess_return = self.portfolio_expected_return - risk_free_rate
        return excess_return / self.portfolio_volatility
    
    def get_weights(self) -> Dict[str, float]:
        """
        Returns the current portfolio weights as a dictionary.
        
        Returns:
        - A dictionary of ticker weights.
        """
        return {ticker: self.w[i] for i, ticker in enumerate(self.tickers)}

    def get_covariance_matrix(self) -> pd.DataFrame:
        """
        Returns the covariance matrix as a DataFrame with tickers as row and column names.
        
        Returns:
        - A pandas DataFrame representing the covariance matrix.
        """
        return pd.DataFrame(self.Sigma, index=self.tickers, columns=self.tickers)

    def get_expected_returns_and_sigmas(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns the expected returns and sigmas (standard deviations) as dictionaries.
        
        Returns:
        - A tuple of two dictionaries:
            1. Expected returns for each ticker.
            2. Standard deviations (sigmas) for each ticker.
        """
        expected_returns_dict = {ticker: self.expected_returns[i] for i, ticker in enumerate(self.tickers)}
        sigmas_dict = {ticker: self.sigmas[i] for i, ticker in enumerate(self.tickers)}
        return expected_returns_dict, sigmas_dict

    def _optimize_mean_variance(self) -> Dict[str, float]:
        """
        Optimizes portfolio weights for mean-variance optimization (minimum volatility for given returns).
        
        Returns:
        - A dictionary of optimized weights for each ticker.
        """
        # Objective function: Minimize portfolio variance
        def objective(w: np.ndarray) -> float:
            return np.dot(w.T, np.dot(self.Sigma, w))  # Portfolio variance

        # Constraints: Weights sum to 1
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(len(self.tickers))]  # Non-negative weights

        # Optimize
        result = minimize(
            objective, self.w, method="SLSQP", constraints=constraints, bounds=bounds
        )
        if not result.success:
            raise ValueError("Mean-Variance Optimization failed: " + result.message)

        optimized_weights = result.x
        return {ticker: optimized_weights[i] for i, ticker in enumerate(self.tickers)}

    def _optimize_max_sharpe(self) -> Dict[str, float]:
        """
        Optimizes portfolio weights for maximum Sharpe ratio.
        
        Returns:
        - A dictionary of optimized weights for each ticker.
        """
        # Objective function: Maximize Sharpe ratio (minimize negative Sharpe)
        def objective(w: np.ndarray) -> float:
            portfolio_return = np.dot(w, self.expected_returns)
            portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(self.Sigma, w)))
            sharpe_ratio = (portfolio_return - self.risk_free) / portfolio_volatility
            return -sharpe_ratio  # Maximize Sharpe ratio (minimize negative Sharpe)

        # Constraints: Weights sum to 1
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(len(self.tickers))]  # Non-negative weights

        # Optimize
        result = minimize(
            objective, self.w, method="SLSQP", constraints=constraints, bounds=bounds
        )
        if not result.success:
            raise ValueError("Maximum Sharpe Ratio Optimization failed: " + result.message)

        optimized_weights = result.x
        return {ticker: optimized_weights[i] for i, ticker in enumerate(self.tickers)}

    def _optimize_max_div(self) -> Dict[str, float]:
        """
        Optimizes portfolio weights for maximum diversification ratio.
        
        Returns:
        - A dictionary of optimized weights for each ticker.
        """
        # Objective function: Negative diversification ratio
        def objective(w: np.ndarray) -> float:
            weighted_sigma = np.dot(w, self.sigmas)
            portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(self.Sigma, w)))
            diversification_ratio = weighted_sigma / portfolio_volatility
            return -diversification_ratio  # Minimize negative DR

        # Constraints: Weights sum to 1
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(len(self.tickers))]  # Non-negative weights

        # Optimize
        result = minimize(
            objective, self.w, method="SLSQP", constraints=constraints, bounds=bounds
        )
        if not result.success:
            raise ValueError("Maximum Diversification Ratio Optimization failed: " + result.message)

        optimized_weights = result.x
        return {ticker: optimized_weights[i] for i, ticker in enumerate(self.tickers)}

    def optimize_weights(self, method: str = "max_div", update_weights: bool = False) -> Dict[str, float]:
        """
        Optimizes portfolio weights based on the specified method.
        
        Parameters:
        - method: Optimization method. Can be "max_div", "mean_variance", or "max_sharpe".
        - update_weights: If True, updates the portfolio weights with the optimized weights.
        
        Returns:
        - A dictionary of optimized weights for each ticker.
        """
        if method == "max_div":
            optimized_weights = self._optimize_max_div()
        elif method == "mean_variance":
            optimized_weights = self._optimize_mean_variance()
        elif method == "max_sharpe":
            optimized_weights = self._optimize_max_sharpe()
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        if update_weights:
            self.w = np.array([optimized_weights[ticker] for ticker in self.tickers])
            self.portfolio_expected_return = np.dot(self.w, self.expected_returns)
            self.portfolio_volatility = np.sqrt(np.dot(self.w.T, np.dot(self.Sigma, self.w)))
            self.diversification_ratio = self._calculate_diversification_ratio()
            self.sharpe_ratio = self._calculate_sharpe_ratio(risk_free_rate=self.risk_free)

        return optimized_weights
    
    def update_portfolio(self, 
                        add_tickers: Optional[Dict[str, np.ndarray]] = None, 
                        remove_tickers: Optional[List[str]] = None, 
                        new_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Updates the portfolio by adding and/or removing tickers and recalculates attributes.
        
        Parameters:
        - add_tickers: A dictionary with new tickers as keys and their corresponding return vectors as values.
        - remove_tickers: A list of tickers to remove from the portfolio.
        - new_weights: A dictionary of desired weights for the new portfolio.
        """
        # Step 1: Handle removing tickers
        if remove_tickers:
            for ticker in remove_tickers:
                if ticker in self.raw_returns:
                    del self.raw_returns[ticker]
                else:
                    print(f"Ticker {ticker} not found in the portfolio.")

        # Step 2: Handle adding tickers
        if add_tickers:
            for ticker, returns in add_tickers.items():
                if len(returns) != len(next(iter(self.raw_returns.values()))):
                    raise ValueError(f"Dimension mismatch for ticker {ticker}. "
                                    f"Expected length {len(next(iter(self.raw_returns.values())))}.")
                self.raw_returns[ticker] = returns  # Replace or add the ticker

        # Step 3: Recalculate attributes
        self.tickers = list(self.raw_returns.keys())
        self.returns_matrix = np.column_stack([self.raw_returns[ticker] for ticker in self.tickers])
        self.Sigma = np.cov(self.returns_matrix, rowvar=False)
        self.sigmas = np.sqrt(np.diag(self.Sigma))
        self.expected_returns = self.returns_matrix.mean(axis=0)

        # Step 4: Handle new weights or normalize existing weights
        if new_weights:
            if set(new_weights.keys()) != set(self.tickers):
                raise ValueError("new_weights must match the tickers in the updated portfolio.")
            total_weight = sum(new_weights.values())
            self.w = np.array([new_weights[ticker] / total_weight for ticker in self.tickers])
        else:
            # Proportionally normalize existing weights to account for changes
            current_weights = {ticker: self.w[i] for i, ticker in enumerate(self.tickers) if ticker in self.raw_returns}
            total_weight = sum(current_weights.values())
            self.w = np.array([current_weights.get(ticker, 0) / total_weight for ticker in self.tickers])

        # Step 5: Recalculate portfolio statistics
        self.portfolio_expected_return = np.dot(self.w, self.expected_returns)
        self.portfolio_volatility = np.sqrt(np.dot(self.w.T, np.dot(self.Sigma, self.w)))
        self.diversification_ratio = self._calculate_diversification_ratio()

    def summarize_portfolio(self) -> None:
        """
        Prints a summary of the portfolio including tickers, weights, and portfolio statistics.
        """
        print("Portfolio Summary:")
        print("Tickers:", self.tickers)
        print("Weights:", self.get_weights())
        print("Portfolio Expected Return:", self.portfolio_expected_return)
        print("Portfolio Volatility:", self.portfolio_volatility)
        print("Portfolio Diversification Ratio:", self.diversification_ratio)
        print("Portfolio Sharpe Ratio:", self.sharpe_ratio)

    def copy_portfolio(self) -> "Portfolio":
        """
        Creates a deep copy of the current portfolio.

        Returns:
        - A new Portfolio object with the same data, frequency, and weights as the current portfolio.
        """
        return Portfolio(
            returns_dict={ticker: np.copy(data) for ticker, data in self.raw_returns.items()},
            frequency=self.frequency,
            weights={ticker: weight for ticker, weight in zip(self.tickers, self.w)}, 
            risk_free=self.risk_free
        )

    def visualize_portfolio_distribution(self) -> None:
        """
        Visualizes the portfolio distribution using a pie chart.
        """
        plt.figure(figsize=(10, 6))
        plt.pie(self.w, labels=self.tickers, autopct="%1.1f%%", startangle=140)
        plt.axis("equal")
        plt.title("Portfolio Distribution")
        plt.show()

"""
EXAMPLE: 

### Initialization 
# Example returns for three stocks (numpy arrays of simulated returns)
returns_dict = {
    "AAPL": np.random.normal(0.001, 0.02, 1000),  # Apple
    "GOOGL": np.random.normal(0.0012, 0.025, 1000),  # Google
    "MSFT": np.random.normal(0.0008, 0.015, 1000)   # Microsoft
}

# Example weights (optional)
weights = {
    "AAPL": 0.4,
    "GOOGL": 0.4,
    "MSFT": 0.2
}

# Frequency of returns
frequency = "daily"

# Initialize the Portfolio object
portfolio = Portfolio(returns_dict=returns_dict, frequency=frequency, weights=weights)

#######################################################################################

### Accessing Attributes 
# Print the tickers in the portfolio
print("Tickers:", portfolio.tickers)

# Print the raw returns dictionary
print("Raw Returns:")
for ticker, returns in portfolio.raw_returns.items():
    print(f"{ticker}: {returns[:5]}...")  # Print first 5 returns for each ticker

# Print the covariance matrix
print("Covariance Matrix (Sigma):\n", portfolio.Sigma)

# Print the diagonal of the covariance matrix (sigmas)
print("Volatilities (sigmas):", portfolio.sigmas)

# Portfolio statistics
print("Portfolio Expected Return:", portfolio.portfolio_expected_return)
print("Portfolio Volatility:", portfolio.portfolio_volatility)
print("Portfolio Diversification Ratio:", portfolio.diversification_ratio)


########################################################################################

### Calling Methods 

# Get covariance matrix as a DataFrame
cov_matrix_df = portfolio.get_covariance_matrix()
print("Covariance Matrix DataFrame:\n", cov_matrix_df)

# Get expected returns and volatilities (sigmas) as dictionaries
expected_returns, sigmas = portfolio.get_expected_returns_and_sigmas()
print("Expected Returns:", expected_returns)
print("Volatilities (sigmas):", sigmas)

# Get initial weights
init_weights = portfolio.get_weights()

# Optimize weights for maximum diversification
optimized_weights = portfolio.optimize_weights(method="max_div", update_weights=False)
print("Optimized Weights:", optimized_weights)

# Update weights in the portfolio object after optimization
portfolio.optimize_weights(method="max_div", update_weights=True)

# Show weights before and after optimization 
print("Initial Weights:", init_weights)
print("Updated Weights:", portfolio.get_weights())

# Updated portfolio statistics after optimization
print("Updated Portfolio Expected Return:", portfolio.portfolio_expected_return)
print("Updated Portfolio Volatility:", portfolio.portfolio_volatility)
print("Updated Portfolio Diversification Ratio:", portfolio.diversification_ratio)
"""
