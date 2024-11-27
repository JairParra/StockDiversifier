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
import time
import random
import warnings
import yfinance as yf
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

# Data Manipulation & Visualiztion  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optimization & Distance Metrics 
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

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

            # Wait one second to avoid hitting the Yahoo Finance API rate limit
            time.sleep(1)
        
        except Exception as e:
            print(f"Failed to fetch or process data for {ticker}: {str(e)}")
    
    return returns_dict


def diversify_betavae_portfolio(
    portfolio, 
    portfolio_embeddings, 
    all_returns, 
    all_stock_embeddings, 
    num_iter=20, 
    top_N=5, 
    distance_type="euclidean", 
    optim_algorithm="max_div", 
    verbose = 1
):
    """
    Diversify a portfolio based on embedding similarity and maximize diversification ratio.
    
    Parameters:
    - portfolio: Portfolio object.
    - portfolio_embeddings: Dictionary of portfolio ticker embeddings.
    - all_returns: Dictionary of tickers and their corresponding return vectors.
    - all_stock_embeddings: Dictionary of all stock embeddings.
    - num_iter: Maximum number of iterations.
    - top_N: Number of most dissimilar stocks to consider for replacement.
    - distance_type: Type of distance metric ("euclidean" or "cosine").
    - optim_algorithm: Optimization algorithm to use (e.g., "max_div", "mean_variance").
    
    Returns:
    - updated_portfolio: Updated Portfolio object.
    - diversification_history: List of diversification ratios at each step.
    - swap_log: Dictionary of removed tickers and their replacements.
    """
    # Deep copy the portfolio
    updated_portfolio = portfolio.copy_portfolio()
    
    # Ensure the portfolio is optimized
    updated_portfolio.optimize_weights(method=optim_algorithm, update_weights=True)

    # Calculate the initial diversification ratio
    initial_diversification_ratio = updated_portfolio.diversification_ratio
    current_diversification_ratio = initial_diversification_ratio
    
    # Exclude portfolio tickers from all_stock_embeddings
    excluded_tickers = set(updated_portfolio.tickers)
    available_tickers = {ticker: embedding for ticker, embedding in all_stock_embeddings.items() 
                         if ticker not in excluded_tickers}
    
    diversification_history = [initial_diversification_ratio]
    swap_log = {}
    
    if verbose >= 1: 
        print(f"Initial Diversification Ratio: {initial_diversification_ratio}")
    
    for _ in tqdm(range(num_iter), desc="Diversification Iterations"):
        # 3. Find the two most similar stocks in the portfolio
        tickers = list(portfolio_embeddings.keys())
        embeddings = np.array(list(portfolio_embeddings.values()))
        pairwise_distances = cdist(embeddings, embeddings, metric=distance_type)
        np.fill_diagonal(pairwise_distances, np.inf)  # Ignore self-similarity
        
        # Find the indices of the most similar pair
        i, j = np.unravel_index(np.argmin(pairwise_distances), pairwise_distances.shape)
        most_similar_pair = [tickers[i], tickers[j]]
        
        # 4. Pick one of these tickers at random
        ticker_to_replace = random.choice(most_similar_pair)
        ticker_embedding = portfolio_embeddings[ticker_to_replace]
        
        # Find top_N most dissimilar stocks from the universe
        all_embeddings = np.array(list(available_tickers.values()))
        distances_to_ticker = cdist([ticker_embedding], all_embeddings, metric=distance_type).flatten()
        top_dissimilar_indices = np.argsort(distances_to_ticker)[-top_N:]
        top_dissimilar_tickers = [list(available_tickers.keys())[idx] for idx in top_dissimilar_indices]
        
        # Choose one of these tickers at random
        replacement_ticker = random.choice(top_dissimilar_tickers)
        replacement_returns = all_returns[replacement_ticker]  # Correctly fetch return vectors
        
        # Temporarily create a new copy of the portfolio for the swap
        temp_portfolio = updated_portfolio.copy_portfolio()
        
        try:
            # Apply the potential swap
            add_tickers = {replacement_ticker: replacement_returns}
            remove_tickers = [ticker_to_replace]
            temp_portfolio.update_portfolio(add_tickers=add_tickers, remove_tickers=remove_tickers)
            temp_portfolio.optimize_weights(method=optim_algorithm, update_weights=True)
            
            # Calculate the new diversification ratio
            new_diversification_ratio = temp_portfolio.diversification_ratio
            
            # Accept or reject the swap
            if new_diversification_ratio > current_diversification_ratio:
                # Accept the swap
                updated_portfolio = temp_portfolio
                current_diversification_ratio = new_diversification_ratio
                diversification_history.append(current_diversification_ratio)
                excluded_tickers.add(ticker_to_replace)
                available_tickers.pop(replacement_ticker)
                swap_log[ticker_to_replace] = replacement_ticker
                
                if verbose >= 1: 
                    print(f"Accepted Swap: {ticker_to_replace} -> {replacement_ticker}")
                    print(f"New Diversification Ratio: {new_diversification_ratio}")
                
                # Update portfolio embeddings
                portfolio_embeddings.pop(ticker_to_replace)
                portfolio_embeddings[replacement_ticker] = all_stock_embeddings[replacement_ticker]
            else:
                if verbose >= 2: 
                    print(f"Rejected Swap: {ticker_to_replace} -> {replacement_ticker}")
        except Exception as e:
            print(f"Error during swap attempt: {e}")
            print(f"Portfolio state after error: {updated_portfolio.raw_returns.keys()}")
            continue

    return updated_portfolio, diversification_history, swap_log


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
        self.raw_returns = {ticker: self.returns_matrix[:, i] for i, ticker in enumerate(self.tickers)}

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

    # def visualize_portfolio_distribution(self) -> None:
    #     """
    #     Visualizes the portfolio distribution using a pie chart.
    #     """
    #     plt.figure(figsize=(10, 6))
    #     plt.pie(self.w, labels=self.tickers, autopct="%1.1f%%", startangle=140)
    #     plt.axis("equal")
    #     plt.title("Portfolio Distribution")
    #     plt.show()


    def visualize_portfolio_distribution(self, pname="", figsize=(10, 6)) -> None:
        """
        Visualizes the portfolio distribution using a pie chart with improved label positioning.
        
        Parameters:
        - figsize: Size of the figure.
        """
        plt.figure(figsize=figsize)
        
        # Create the pie chart with label positions set to None initially
        wedges, texts, autotexts = plt.pie(
            self.w, 
            labels=None,  # Disable labels initially
            autopct="%1.1f%%", 
            startangle=140
        )

        # Adjust label positions manually to avoid overlap
        for wedge, label in zip(wedges, self.tickers):
            angle = (wedge.theta2 + wedge.theta1) / 2  # Compute the midpoint angle
            x = 1.1 * np.cos(np.radians(angle))       # Adjust x-coordinate for label
            y = 1.1 * np.sin(np.radians(angle))       # Adjust y-coordinate for label
            plt.text(
                x, y, label, 
                ha="center", va="center", 
                fontsize=9, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
            )
        
        # Improve layout
        plt.axis("equal")  # Equal aspect ratio ensures the pie is circular
        plt.title(f"{pname}Portfolio Distribution")
        plt.tight_layout()
        plt.show()


    # def visualize_portfolio_distribution(self, kind="pie", show_legend=True, figsize=(12, 8), max_labels=15) -> None:
    #     """
    #     Visualizes the portfolio distribution using a pie chart or bar chart.
        
    #     Parameters:
    #     - kind: Type of chart to plot. Can be "pie" or "bar".
    #     - show_legend: Whether to display a legend (for pie chart).
    #     - figsize: Size of the figure.
    #     - max_labels: Maximum number of labels to display directly on the chart. 
    #                 For pie charts, excess labels are aggregated into 'Others'.
    #                 For bar charts, all stocks are shown regardless.
    #     """
    #     # Sort weights and labels for clarity
    #     sorted_weights, sorted_tickers = zip(
    #         *sorted(zip(self.w, self.tickers), reverse=True)
    #     )

    #     # For equal weights, show all stocks
    #     if np.allclose(sorted_weights, np.full_like(sorted_weights, 1 / len(sorted_weights))):
    #         max_labels = len(sorted_weights)

    #     # Aggregate small weights for pie chart
    #     if kind == "pie" and len(sorted_weights) > max_labels:
    #         displayed_weights = list(sorted_weights[:max_labels])
    #         displayed_tickers = list(sorted_tickers[:max_labels])
    #         other_weight = sum(sorted_weights[max_labels:])
    #         displayed_weights.append(other_weight)
    #         displayed_tickers.append("Others")
    #     else:
    #         displayed_weights = sorted_weights
    #         displayed_tickers = sorted_tickers

    #     if kind == "pie":
    #         # Plot pie chart
    #         plt.figure(figsize=figsize)
    #         explode = [0.1 if i == 0 else 0 for i in range(len(displayed_weights))]

    #         wedges, texts, autotexts = plt.pie(
    #             displayed_weights,
    #             labels=None if show_legend else displayed_tickers,
    #             autopct="%1.1f%%",
    #             startangle=90,
    #             explode=explode,
    #             textprops=dict(color="black"),
    #         )

    #         # Add a legend if enabled
    #         if show_legend:
    #             plt.legend(
    #                 loc="upper left",
    #                 labels=[f"{ticker}: {weight:.2%}" for ticker, weight in zip(displayed_tickers, displayed_weights)],
    #                 bbox_to_anchor=(1, 0.5),
    #             )

    #         # Improve layout
    #         plt.axis("equal")  # Equal aspect ratio ensures the pie is circular
    #         plt.title("Portfolio Distribution (Pie Chart)")
    #         plt.tight_layout()
    #         plt.show()

    #     elif kind == "bar":
    #         # Plot bar chart
    #         plt.figure(figsize=figsize)
    #         plt.bar(displayed_tickers, displayed_weights, color="skyblue")
    #         plt.xlabel("Tickers")
    #         plt.ylabel("Weights")
    #         plt.title("Portfolio Distribution (Bar Chart)")
    #         plt.xticks(rotation=45, ha="right")
    #         plt.tight_layout()
    #         plt.show()

    #     else:
    #         raise ValueError(f"Unknown chart type: {kind}. Choose 'pie' or 'bar'.")

