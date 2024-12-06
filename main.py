"""
main.py
    This script contains calls to all the main functions to fetch the data, 
    train and evaluate the model, and then save the model, and call the recommender 
    function for diversification. 

@author: Hair Parra
@author: Olivier Makuch
"""

################## 
### 1. Imports ###
##################

# General 
import os 
import json 
import pprint
import logging 
import argparse
import datetime
from tqdm import tqdm 

# Data Science
import optuna 
import numpy as np 
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# PyTorch
import torch

# Custom Classes 
from src.beta_vae import Encoder
from src.beta_vae import Decoder
from src.beta_vae import BetaVAE
from src.pca_enc import PCAEncoder
from src.portfolio import Portfolio

# Custom Util Functions
from src.utils import serialize_ndarray 
from src.utils import deserialize_ndarray
from src.data_fetching import scrape_sp500_wikipedia
from src.data_fetching import fetch_stock_data
from src.data_fetching import prepare_data_for_vae

# Beta VAE Functions
from src.beta_vae import create_data_loaders
from src.beta_vae import objective
from src.beta_vae import train_beta_vae
from src.beta_vae import generate_embeddings_dict as get_vae_embeddings

# PCA Functions
from src.pca_enc import generate_embeddings_dict as get_pca_embeddings

# Portfolio Functions
from src.portfolio import fetch_and_calculate_returns
from src.portfolio import diversify_betavae_portfolio


###########################
### 2. Script Arguments ###
###########################

# PATH variables
DATA_RAW_DIR = "data_raw"
DATA_CLEAN_DIR = "data_clean"
MODEL_DIR = "models"
CONFIG_DIR = "config"
LOGS_DIR = "logs"

MODEL_PATH = os.path.join(MODEL_DIR, "beta_vae.pth") 
OUTPUT_PATH = os.path.join(DATA_CLEAN_DIR, "recommendations.csv") 
STOCK_DATA_PATH = os.path.join(DATA_RAW_DIR, "stock_data.csv")
RETURNS_DATA_PATH = os.path.join(DATA_RAW_DIR, "all_returns.json")
BEST_PARAMS_PATH = os.path.join(CONFIG_DIR, "best_params.json") 
EXPERIMENTS_LOG_PATH = os.path.join(LOGS_DIR, "experiments_results.csv")
SP500_DATA_PATH = os.path.join(DATA_RAW_DIR, "sp500_data.csv")

# default script parameters
RETRAIN = False
REFETCH = False
LOAD_RETURNS = False
verbose = 1

# default experiment configurations 
NUM_PORTFOLIOS = 20
NUM_INITIAL_TICKERS = 10
OPTIM_METHOD = "max_div"

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Modify default configurations')

# Add arguments for each configuration option
parser.add_argument('--refetch', action='store_true', default=True, help='Refetch the loaded sp500 data')
parser.add_argument('--retrain', action='store_true', default=True, help='Retrain the model')
parser.add_argument('--load_returns', action='store_true', default=True, help='Refetch returns data for all stocks')
parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
parser.add_argument('--recoms_filename', type=str, default=None, help='Custom name for the output recommendation file')

# Add arguments for the default experiment configurations
parser.add_argument('--num_portfolios', type=int, default=NUM_PORTFOLIOS, help='Number of portfolios to generate')
parser.add_argument('--num_initial_tickers', type=int, default=NUM_INITIAL_TICKERS, help='Number of initial tickers in the portfolio')
parser.add_argument('--optim_method', type=str, default=OPTIM_METHOD, help='Optimization method for the portfolio')

# Parse the command line arguments
args = parser.parse_args()

# Update the default configurations with the command line arguments
REFETCH = args.refetch
RETRAIN = args.retrain
LOAD_RETURNS = args.load_returns
verbose = args.verbose

# Update the default experiment configurations with the command line arguments
NUM_PORTFOLIOS = args.num_portfolios
NUM_INITIAL_TICKERS = args.num_initial_tickers
OPTIM_METHOD = args.optim_method

# Update the output path if a custom filename is provided
OUTPUT_PATH = f"data_clean/{args.recoms_filename}.csv" if args.recoms_filename is not None else OUTPUT_PATH

#########################
### 3. Configurations ###
#########################

# Set pprint options with indent 4
pprint = pprint.PrettyPrinter(indent=4).pprint

# Change number of display columns
pd.options.display.max_columns = None

# Configure logging
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"logs/example_{current_datetime}.log"
logging.basicConfig(filename=filename, level=logging.INFO)

# Additional tickers 
CUSTOM_TICKERS = [
    'PLTR',  # Palantir Technologies Inc. (Technology)
    'COIN',  # Coinbase Global, Inc. (Financials)
    'WDAY',  # Workday, Inc. (Technology)
    'TTD',   # The Trade Desk, Inc. (Technology)
    'APO',   # Apollo Global Management, Inc. (Financials)
    'MELI',  # MercadoLibre, Inc. (Consumer Discretionary)
    'NVO',   # Novo Nordisk A/S (Healthcare)
    'ICON',  # Icon PLC (Healthcare)
    'TSM',   # Taiwan Semiconductor Manufacturing Company Limited (Technology)
    'SKX',   # Skechers U.S.A., Inc. (Consumer Discretionary)
    'BAYRY', # Bayer AG (Healthcare)
    'DISCK', # Discovery, Inc. (Communication Services)
    'FNV',   # Franco-Nevada Corporation (Materials)
    'SHOP',  # Shopify Inc. (Technology)
    'SQ',    # Square, Inc. (Technology)
    'UBER',  # Uber Technologies, Inc. (Industrials)
    'ZM',    # Zoom Video Communications, Inc. (Technology)
    'TWLO',  # Twilio Inc. (Technology)
    'MRNA',  # Moderna, Inc. (Healthcare)
    'WDAY',  # Workday, Inc. (Technology)
    'DOCU',  # DocuSign, Inc. (Technology)
    'VEEV',  # Veeva Systems Inc. (Healthcare)
    'LULU',  # Lululemon Athletica Inc. (Consumer Discretionary)
    'ROKU',  # Roku, Inc. (Communication Services)
    'CRWD',  # CrowdStrike Holdings, Inc. (Technology)
    'SNOW',  # Snowflake Inc. (Technology)
    'NET',   # Cloudflare, Inc. (Technology)
    'PINS',  # Pinterest, Inc. (Communication Services)
    'ETSY',  # Etsy, Inc. (Consumer Discretionary)
    'SPOT',   # Spotify Technology S.A. (Communication Services)
    "AAPL", # Apple Inc. (Technology)
]

#######################
### 4. Driver Code  ###
#######################

if __name__ == "__main__": 

    # Show all the arguments the script ascall with to the user 
    print(f"Script with called with arguments: {vars(args)}")

    ########################
    ### 4.1 Scrape SP500 ###
    ########################

    print("1. Fetching Data")

    if REFETCH:
        print("Refetching S&P 500 Data")

        # Scrape S&P 500 companies names, tikers and industris from Wikipedia
        sp500_df = scrape_sp500_wikipedia()

        # Save the data to a file
        sp500_df.to_csv(SP500_DATA_PATH, index=False)
    else: 
        # Load the data from the file
        sp500_df = pd.read_csv(SP500_DATA_PATH)

    #########################################################
    ### 4.2 Fetch or Load (Aggregated) Raw Stock Features ###
    #########################################################

    print("2. Fetching Stock Data Features")
    
    if REFETCH:
        # Fetch data for all sp500 + custom tickers
        stock_data, sector_mapping, industry_mapping = fetch_stock_data(sp500_df, CUSTOM_TICKERS, 
                                                                        period = "1y", interval = "1wk",
                                                                        savepath= STOCK_DATA_PATH) # Fetch data
    else: 
        # read data from file
        stock_data = pd.read_csv(STOCK_DATA_PATH, index_col=0).reset_index()


    ####################################################
    ### 4.3 Fetch or Load (Historical) Stock Returns ###
    ####################################################

    print("3. Fetching Stock Returns")

    if LOAD_RETURNS:
        # Load the returns from the file
        with open(RETURNS_DATA_PATH, "r") as f:
            all_returns = json.load(f)

        # Deserialize loaded data
        all_returns = deserialize_ndarray(all_returns)

    else: 
        # Fetch and calculate returns for the all tickers in the S&P 500 and additional tickers
        all_tickers = stock_data["Ticker"].values.tolist()  # All S&P 500 tickers
        print(f"number of tickers: {len(all_tickers)}")

        # Fetch and calculate returns for all tickers
        all_returns = fetch_and_calculate_returns(all_tickers, period="1y", interval="1wk", price_column="Close")

        # Find the mode of the lengths of the returns among all tickers
        return_lengths = [len(returns) for returns in all_returns.values()]
        return_length_mode = max(set(return_lengths), key=return_lengths.count)

        # Ensure to keep only the returns with the mode length and report the discarded tickers 
        discarded_tickers = [ticker for ticker, returns in all_returns.items() if len(returns) != return_length_mode]
        all_returns = {ticker: returns for ticker, returns in all_returns.items() if len(returns) == return_length_mode}
        print(f"Discarded tickers with inconsistent return lengths: {discarded_tickers}")

        # Remove from stock_data the tickers with inconsistent return lengths
        stock_data = stock_data[stock_data["Ticker"].isin(all_returns.keys())].reset_index().drop(columns=["index"])

        print("Total number of tickers with consistent return lengths:", len(all_returns))
        print("stock_date shape:", stock_data.shape)

        # Save the dictionary with the returns to a file using json 
        with open(RETURNS_DATA_PATH, "w") as f:
            print(f"Saving returns to {f.name}")
            json.dump(serialize_ndarray(all_returns), f, indent=4)

    ###############################################
    ### 4.4 Prepare Data To Train the Beta-VAE  ###
    ###############################################

    print("4. Preparing Data to Fit the Beta-VAE")

    # Prepare data for VAE
    stock_data_vae = prepare_data_for_vae(stock_data) 

    #############################################################################################################################
    #############################################################################################################################


    ############################################
    ### 4.5 Preparing Data for the Beta-VAE  ###
    ############################################

    # Normalize the stock data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(stock_data_vae)

    # Convert the normalized data to PyTorch tensors
    tensor_data = torch.tensor(normalized_data, dtype=torch.float32)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(tensor_data)
    
    #################################
    ### 4.6 Beta-VAE Hypertuning  ###
    #################################

    ## Either train and save the model to models/ or reload from there

    # Check if the best_params file exists
    if not RETRAIN and os.path.exists(BEST_PARAMS_PATH):
        # Load the best parameters from the file
        with open(BEST_PARAMS_PATH, "r") as f:
            best_params = json.load(f)
        print("Loaded best parameters from file:", best_params)
        
    else:
        # Delete previous best_params file if exists
        if os.path.exists(BEST_PARAMS_PATH):
            os.remove(BEST_PARAMS_PATH)

        # File doesn't exist, run the Optuna study
        print("Best parameters file not found. Running Optuna study...")
        
        # Assuming `normalized_data` is the prepared dataset
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, normalized_data, verbose=False), n_trials=100)

        # Get the best hyperparameters
        best_params = study.best_params
        print("Best hyperparameters found:", best_params)

        # Save the best parameters to a JSON file
        os.makedirs(os.path.dirname(BEST_PARAMS_PATH), exist_ok=True)  # Create directory if it doesn't exist
        with open(BEST_PARAMS_PATH, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"Best parameters saved to {BEST_PARAMS_PATH}")


    ##########################################
    ### 4.7 Beta-VAE Final Model Training  ###
    ##########################################

    if RETRAIN:
        # Delete previous model file
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

    # Check if the model file exists
    if not RETRAIN and os.path.exists(MODEL_PATH):
        print(f"Model found at {MODEL_PATH}. Loading the model...")

        # Load model state and metadata
        model_metadata = torch.load(MODEL_PATH)

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
        print(f"Training a new model...")

        # Assuming you use the best hyperparameters from Optuna
        latent_dim = best_params['latent_dim']
        beta = best_params['beta']
        learning_rate = best_params['learning_rate']
        batch_size = best_params['batch_size']
        input_dim = stock_data_vae.shape[1]

        # Recreate DataLoaders with the best batch sizes
        train_loader, val_loader, test_loader = create_data_loaders(tensor_data, batch_size=batch_size)

        # Initialize and train the model
        beta_vae = BetaVAE(input_dim=input_dim, latent_dim=latent_dim, beta=beta)
        train_beta_vae(beta_vae, train_loader, val_loader, num_epochs=50, learning_rate=learning_rate)

        # Save the trained model with metadata
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Ensure the directory exists
        model_metadata = {
            "model_state": beta_vae.state_dict(),
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "beta": beta
        }
        torch.save(model_metadata, MODEL_PATH)
        print(f"Model trained and saved to {MODEL_PATH}.")


    ##########################
    ### 4.8 PCA Encodings  ###
    ##########################

    # Initialize PCA encoder with same dimension as the latent dim of the Beta-VAE
    pca_encoder = PCAEncoder(data=normalized_data, n_components=best_params["latent_dim"])

    # Initialize another PCA encoder with optimal number of components 
    pca_encoder_tuned = PCAEncoder(data=normalized_data)
    pca_encoder_tuned.autotune_num_components(explained_var=0.9) # Autotune to 90% explained variance

    #############################################################################################################################
    #############################################################################################################################\

    #####################################################
    ### 4.9 Extract Embeddings for Universe of Stocks ###
    #####################################################

    # Generate embeddings for the stock data
    stock_embeddings = get_vae_embeddings(stock_data, scaler, beta_vae, ticker_col="Ticker")
    pca_stock_embeddings = get_pca_embeddings(stock_data, scaler, pca_encoder, ticker_col="Ticker")
    pca_stock_embeddings_tuned = get_pca_embeddings(stock_data, scaler, pca_encoder_tuned,ticker_col="Ticker")

    ################################################
    ### 4.8 Beta-VAE Diversification Experiments ###
    ################################################

    # Initialize storage for portfolio results
    portfolio_results = []
    portfolio_tickers = {}

    # Define number of iterations for diversification process
    optim_method = "max_div"

    for i in tqdm(range(1, NUM_PORTFOLIOS + 1), desc="Portfolio Diversification Experiments..."):
        # Step 1: Generate random portfolio
        random_tickers = np.random.choice(list(all_returns.keys()), NUM_INITIAL_TICKERS, replace=False)
        random_returns = {ticker: all_returns[ticker] for ticker in random_tickers}
        random_portfolio = Portfolio(returns_dict=random_returns, frequency="weekly")
        
        # Optimize initial portfolio
        random_portfolio.optimize_weights(method=optim_method, update_weights=True)
        
        # Record initial diversification ratio (DR) and Sharpe ratio (SR)
        initial_dr = random_portfolio.diversification_ratio
        initial_sr = random_portfolio.sharpe_ratio
        
        # Initialize results container for this portfolio
        portfolio_experiments = []
        
        # Define embedding methods and corresponding data
        methods = [
            ("Beta-VAE", stock_embeddings),
            ("PCA (Latent Dim)", pca_stock_embeddings),
            ("PCA (90% Var)", pca_stock_embeddings_tuned)
        ]
        
        for method_name, embeddings in methods:
            # Step 2: Generate portfolio-specific embeddings
            random_portfolio_embeddings = {
                ticker: embeddings[ticker] for ticker in random_portfolio.tickers if ticker in embeddings.keys()
            }
            assert len(random_portfolio_embeddings) == len(random_portfolio.tickers)

            # Step 3: Apply diversification function
            updated_portfolio, diversification_history, swap_log = diversify_betavae_portfolio(
                portfolio=random_portfolio,
                portfolio_embeddings=random_portfolio_embeddings,
                all_returns=all_returns,
                all_stock_embeddings=embeddings,
                num_iter=250,
                top_N=10,
                optim_algorithm=optim_method,
                distance_type="euclidean",
                verbose=False
            )
            
            # Optimize the updated portfolio after swaps
            updated_portfolio.optimize_weights(method=optim_method, update_weights=True)
            
            # Record updated DR and SR
            final_dr = updated_portfolio.diversification_ratio
            final_sr = updated_portfolio.sharpe_ratio
            
            # Store results for this method
            portfolio_experiments.append({
                "Portfolio": f"Portfolio_{i}",
                "Method": method_name,
                "Initial_DR": initial_dr,
                "Final_DR": final_dr,
                "DR_Improvement": round(((final_dr - initial_dr) / initial_dr) * 100, 3),  # pct improvement
                "Initial_SR": initial_sr,
                "Final_SR": final_sr,
                "SR_Improvement": round(((final_sr - initial_sr) / initial_sr) * 100, 3),  # pct improvement
                "Tickers": updated_portfolio.tickers
            })
            
        # Append all results for this portfolio to the overall results
        portfolio_results.extend(portfolio_experiments)

    # Convert results to a DataFrame
    portfolio_results_df = pd.DataFrame(portfolio_results)

    # Sort results by method and DR improvement
    portfolio_results_df = portfolio_results_df.sort_values(by=["Method", "DR_Improvement"], ascending=False)

    ################################################
    ### 4.8 Beta-VAE Diversification Experiments ###
    ################################################

    # Display the table
    print(portfolio_results_df.sort_values(by=["DR_Improvement"], ascending=False))

    # disply sorted by final SR
    print(portfolio_results_df.sort_values(by="SR_Improvement", ascending=False))

    # display sorted by method and DR improvement
    print(portfolio_results_df.sort_values(by=["Method", "DR_Improvement"], ascending=False))

    #############################################
    ### 4.9 More Tables Reported in the Paper ###
    #############################################

    print("#"*100) 
    print("Average Results for all Methods")

    # Compute the average initial and final DR, and average DR improvement for each method
    avg_results = portfolio_results_df.groupby('Method').agg(
        Average_Initial_DR=('Initial_DR', 'mean'),
        Average_Final_DR=('Final_DR', 'mean'),
        Average_DR_Improvement=('DR_Improvement', 'mean')
    ).reset_index()

    # Sort by Average_DR_Improvement descending for clarity
    avg_results = avg_results.sort_values(by="Average_DR_Improvement", ascending=False)

    print(avg_results)

    #############################################

    print("#"*100)
    print("Results for the Beta-VAE method:")

    # Filter for Beta-VAE and sort by DR Improvement
    beta_vae_top = portfolio_results_df[portfolio_results_df['Method'] == 'Beta-VAE'] \
        .sort_values(by='DR_Improvement', ascending=False).head(3)

    print(beta_vae_top[['Portfolio', 'Initial_DR', 'Final_DR', 'DR_Improvement', 'Initial_SR', 'Final_SR', 'SR_Improvement']])

    #############################################

    print("#"*100)
    print("Results for the PCA (Latent Dim) method:")

    # Filter for PCA (Latent Dim) and sort by DR Improvement
    pca_latent_top = portfolio_results_df[portfolio_results_df['Method'] == 'PCA (Latent Dim)'] \
        .sort_values(by='DR_Improvement', ascending=False).head(3)

    print(pca_latent_top[['Portfolio', 'Initial_DR', 'Final_DR', 'DR_Improvement', 'Initial_SR', 'Final_SR', 'SR_Improvement']])

    #############################################
    print("#"*100)
    print("Results for the PCA (90% Var) method:")

    # Filter for PCA (90% Var) and sort by DR Improvement
    pca_90var_top = portfolio_results_df[portfolio_results_df['Method'] == 'PCA (90% Var)'] \
        .sort_values(by='DR_Improvement', ascending=False).head(3)

    print(pca_90var_top[['Portfolio', 'Initial_DR', 'Final_DR', 'DR_Improvement', 'Initial_SR', 'Final_SR', 'SR_Improvement']])

    #############################################
    print("#"*100)

    # Save the table to csv under logs folder
    portfolio_results_df.to_csv(EXPERIMENTS_LOG_PATH, index=False)

