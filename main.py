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


###########################
### 2. Script Arguments ###
###########################

# default configurations 
RETRAIN = True
TRAINING_DATA_DIR = "data_clean/training_data.csv"
MODEL_PATH = "models/beta_vae.pth"
OUTPUT_PATH = "output/recommendations.csv"
verbose = 1

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Modify default configurations')

# Add arguments for each configuration option
parser.add_argument('--retrain', action='store_true', default=True, help='Retrain the model')
parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
parser.add_argument('--custom_output_name', type=str, default=None, help='Custom name for the output file')

# Parse the command line arguments
args = parser.parse_args()

# Update the default configurations with the command line arguments
LOAD_SAMPLE_DATA = args.load_sample_data
RETRAIN = args.retrain
LIMIT_FETCH = args.limit_fetch
verbose = args.verbose
OUTPUT_PATH = f"data_clean/{args.custom_output_name}.csv" if args.custom_output_name is not None else OUTPUT_PATH

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

    print("1. Fetching Data")

    ## Either fetch and save or load the data 
    

    print("2. Training Model") 

    ## Either train and save the model to models/ or reload from there

