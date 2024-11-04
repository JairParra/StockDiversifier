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
import pprint
import logging 
import argparse
import datetime
from tqdm import tqdm 

# Data Science
import numpy as np 
import pandas as pd 

# Custom 
from src.data_fetching import scrape_sp500_wikipedia, fetch_stock_data, prepare_data_for_vae


###########################
### 2. Script Arguments ###
###########################

# default configurations 
RETRAIN = False
REFETCH = False
DATA_RAW_DIR = "data_raw/"
DATA_CLEAN_DIR = "data_clean/"
MODEL_PATH = "models/beta_vae.pth"
OUTPUT_PATH = "data_clean/recommendations.csv"
STOCK_DATA_SAVEPATH = "data_raw/stock_data.csv"
verbose = 1

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Modify default configurations')

# Add arguments for each configuration option
parser.add_argument('--refetch', action='store_true', default=True, help='Refetch the loaded sp500 data')
parser.add_argument('--retrain', action='store_true', default=True, help='Retrain the model')
parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
parser.add_argument('--recoms_filename', type=str, default=None, help='Custom name for the output recommendation file')

# Parse the command line arguments
args = parser.parse_args()

# Update the default configurations with the command line arguments
REFETCH = args.refetch
RETRAIN = args.retrain
verbose = args.verbose
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
TICKERS = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA', 'FB', 'NVDA', 'PYPL', 'ADBE', 'NFLX']

#######################
### 4. Driver Code  ###
#######################

if __name__ == "__main__": 

    ##########################
    ### 4.1 Data Fetching  ###
    ##########################

    # Show all the arguments the script ascall with to the user 
    print(f"Script with called with arguments: {vars(args)}")

    print("1. Fetching Data")

    # sp500_df = scrape_sp500_wikipedia()  # Use the function you created to scrape S&P 500 companies
    # custom_tickers = ['TSLA', 'ZM', 'SNOW']  # Example custom tickers
    # stock_data, sector_mapping, industry_mapping = fetch_stock_data(sp500_df, custom_tickers) # Fetch data
    # stock_data_vae = prepare_data_for_vae(stock_data)  # Prepare data for VAE


    ## Either fetch and save or load the data 
    if REFETCH: 
        print("Refetching data")

        # Scrape S&P 500 companies and save to data_raw
        sp500_df = scrape_sp500_wikipedia()

        # Fetch stock data and save to data_raw
        stock_data, sector_mapping, industry_mapping = fetch_stock_data(sp500_df, TICKERS, savepath=STOCK_DATA_SAVEPATH)

        # Convert the stock data to VAE format 
        stock_data_vae = prepare_data_for_vae(stock_data)

    else: 
        print("Loading data")

        # Load data from data_raw 
        sp500_df = pd.read_csv(os.path.join(DATA_RAW_DIR, 'sp500.csv'))

        # Load pre-prepraed vae_data from data_clean
        stock_data_vae = pd.read_csv(STOCK_DATA_SAVEPATH)

    # # Prepare data for VAE
    stock_data_vae = prepare_data_for_vae(stock_data) 
    
    ###########################
    ### 4.2 Model Training  ###
    ###########################

    ## Either train and save the model to models/ or reload from there

    if RETRAIN:
        print("Retraining model")

        # Train the model 
        pass

    else:
        print("Loading model")

        # Load the model 
        pass

    ############################
    ### 4.3 Recommendations  ###
    ############################

    



