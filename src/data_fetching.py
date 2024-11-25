"""
data_fetching.py
    This script contains functions to scrape the SP500 list from Wikpedia, 
    fetch stock data from Yahoo Finance AP, and then save the data locally to data_clean. 

@author: Hair Parra
"""

################## 
### 1. Imports ###
##################

import time
import warnings
import numpy as np 
import pandas as pd 
import yfinance as yf
from tqdm import tqdm
from typing import List, Tuple
from sklearn.impute import KNNImputer



#########################
### 2. Configurations ###
#########################

SECTORS = ['Energy', 'Industrials', 'Consumer Staples', 'Health Care',
       'Consumer Discretionary', 'Financials', 'Real Estate',
       'Information Technology', 'Utilities', 'Materials',
       'Basic Materials', 'Communication Services', 'Technology',
       'Financial Services', 'Consumer Cyclical', 'Healthcare']

INDUSTRIES = ['Oil & Gas Refining & Marketing', 'Specialty Industrial Machinery', 
              'Household & Personal Products', 'Medical Care Facilities', 'Apparel Retail',
              'Banks - Regional', 'Airlines', 'Auto & Truck Dealerships', 'REIT - Retail',
              'Healthcare Plans', 'Semiconductors', 'Drug Manufacturers - General', 
              'Software - Application', 'Utilities - Regulated Electric', 'Footwear & Accessories',
              'Trucking', 'REIT - Specialty', 'Residential Construction', 
              'Insurance - Property & Casualty', 'Solar', 'Credit Services', 'Banks - Diversified',
              'Building Products & Equipment', 'Utilities - Regulated Water', 'Asset Management', 
              'Specialty Chemicals', 'Travel Services', 'Integrated Freight & Logistics',
              'Chemicals', 'Medical Distribution', 'Oil & Gas Midstream', 'Financial Data & Stock Exchanges',
              'Gold', 'Agricultural Inputs', 'Health Information Services', 'Electronic Components',
              'Internet Content & Information', 'Software - Infrastructure', 'REIT - Industrial',
              'Diagnostics & Research', 'Confectioners', 'Medical Instruments & Supplies',
              'Scientific & Technical Instruments', 'Restaurants', 'Computer Hardware', 'Discount Stores',
              'Entertainment', 'Food Distribution', 'Beverages - Non-Alcoholic', 'REIT - Healthcare Facilities',
              'Capital Markets', 'Packaging & Containers', 'Grocery Stores', 'Packaged Foods',
              'Specialty Business Services', 'REIT - Residential', 'Information Technology Services',
              'Auto Manufacturers', 'Oil & Gas Integrated', 'Insurance Brokers', 'Specialty Retail',
              'Communication Equipment', 'Railroads', 'Farm & Heavy Construction Machinery', 'Personal Services',
              'Medical Devices', 'Biotechnology', 'Oil & Gas Equipment & Services', 'Oil & Gas E&P',
              'REIT - Hotel & Motel', 'REIT - Diversified', 'Aerospace & Defense', 'Resorts & Casinos',
              'Advertising Agencies', 'Real Estate Services', 'Insurance - Life', 'Utilities - Renewable',
              'Building Materials', 'Auto Parts', 'Tobacco', 'Consulting Services', 'REIT - Office',
              'Telecom Services', 'Utilities - Diversified', 'Insurance - Reinsurance', 'Pharmaceutical Retailers', 
              'Semiconductor Equipment & Materials', 'Consumer Electronics', 'Beverages - Brewers',
              'Pollution & Treatment Controls', 'Lodging', 'Industrial Distribution', 'Internet Retail',
              'Apparel Manufacturing', 'Drug Manufacturers - Specialty & Generic', 'Rental & Leasing Services',
              'Home Improvement Retail', 'Luxury Goods', 'Utilities - Independent Power Producers',
              'Insurance - Diversified', 'Engineering & Construction', 'Conglomerates', 'Utilities - Regulated Gas',
              'Tools & Accessories', 'Farm Products', 'Electronic Gaming & Multimedia', 'Steel', 'Waste Management', 
              'Copper', 'Security & Protection Services', 'Furnishings, Fixtures & Appliances', 'Leisure',
              'Electrical Equipment & Parts']

##########################
### 3. Utils Functions ###
##########################

def scrape_sp500_wikipedia():
    """
    Scrapes the list of S&P 500 companies from Wikipedia and returns a DataFrame with selected columns.
    
    This function reads data from the Wikipedia page containing the S&P 500 company listings
    and extracts the 'Symbol', 'Security', and 'GICS Sector' columns.
    
    Returns:
    - DataFrame: A DataFrame containing the Symbol, Security, and GICS Sector columns of S&P 500 companies.
    """

    # URL of the S&P 500 companies list on Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Read all tables from the Wikipedia page
    tables = pd.read_html(url)

    # The first table contains the S&P 500 companies
    sp500_table = tables[0]

    # Select only the Symbol, Security, and GICS Sector columns
    sp500_data = sp500_table[['Symbol', 'Security', 'GICS Sector']]

    return sp500_data

## TEST: Fetch the S&P 500 companies list and print 
# sp500_data = scrape_sp500_wikipedia()
# print(sp500_data.head())


# Function to map unique categories in a specified column to integer values
def encode_categories(dataframe:pd.DataFrame, column_name:str) -> dict:
    """
    Encodes unique categories in a specified DataFrame column as integer values.
    
    This function creates a new column in the DataFrame by mapping each unique value in the specified column
    to a corresponding integer. It also returns a dictionary with the category-to-integer mappings.
    
    Parameters:
    - dataframe (DataFrame): The DataFrame containing the data to be encoded.
    - column_name (str): The name of the column to encode.
    
    Returns:
    - dict: A dictionary where keys are unique categories from the specified column, and values are their
            corresponding integer encoding.
    """

    # Create a dictionary to map each unique category to an integer index
    category_mapping = {category: idx for idx, category in enumerate(dataframe[column_name].unique())}
    
    # Add a new column to the dataframe with the integer encoding of the specified column
    dataframe[column_name + '_encoded'] = dataframe[column_name].map(category_mapping)
    
    # Return the mapping dictionary for reference
    return category_mapping


#########################
### 4. Core Functions ###
#########################

def fetch_stock_data(sp500_tickers_df:pd.DataFrame, 
                     custom_tickers:List[str], 
                     period:str='1y', 
                     interval:str='1wk', 
                     savepath:str=None, 
                     remove_etfs:bool=True, 
                     ) -> Tuple[pd.DataFrame, dict, dict]:

    t0 = time.time()

    # Deactivate warnings for this function only 
    warnings.filterwarnings('ignore')

    # Combine S&P 500 tickers with custom tickers
    tickers = list(sp500_tickers_df['Symbol'].unique()) + custom_tickers
    tickers = list(set(tickers))  # Remove duplicates

    stock_features = []

    for ticker in tqdm(tickers, desc="Fetching stock data..."):
        try:
            # Fetch stock data using Yahoo Finance API
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period=period, interval=interval)
            year_data = stock.history(period="1y", interval="1d")

            if hist_data.empty:
                warnings.warn(f"No data found for ticker {ticker}")
                continue

            if year_data.empty:
                warnings.warn(f"No 1-year data found for ticker {ticker}")
                continue 

            # Calculate basic daily metrics
            open_price = hist_data['Open'].mean()
            close_price = hist_data['Close'].mean()
            high_price = hist_data['High'].max()
            low_price = hist_data['Low'].min()
            last_close = hist_data['Close'][-1]

            # Calculate 52 week high/low
            if not year_data.empty:
                high_52w = year_data['High'].max()
                low_52w = year_data['Low'].min()
                close_1y_ago = year_data['Close'][0]  # Price from 1 year ago
                last_year_return = (last_close - close_1y_ago) / close_1y_ago
            else:
                high_52w, low_52w, last_year_return = np.nan, np.nan, np.nan

            # Volatility measures
            last_month_volatility = hist_data['Close'].pct_change().std() * np.sqrt(len(hist_data))
            if not year_data.empty:
                year_volatility = year_data['Close'].pct_change().std() * np.sqrt(len(year_data))
            else:
                year_volatility = np.nan

            # Meta-data
            is_etf = stock.info.get('quoteType', '') == 'ETF'
            yield_to_date = stock.info.get('yield', np.nan)
            dividend_rate = stock.info.get('dividendRate', np.nan)

            # Look for the sector in the S&P 500 dataframe first
            sp500_sector_row = sp500_tickers_df[sp500_tickers_df['Symbol'] == ticker]
            if not sp500_sector_row.empty:
                sector = sp500_sector_row.iloc[0]['GICS Sector']
            else:
                sector = stock.info.get('sector', 'Other')  # Use 'Other' if sector is unavailable

            industry = stock.info.get('industry', 'Other')
            company_name = stock.info.get('longName', 'N/A')
            market_cap = stock.info.get('marketCap', np.nan)

            # Handling NAs
            yield_to_date = np.nan_to_num(yield_to_date, nan=0)
            dividend_rate = np.nan_to_num(dividend_rate, nan=0)

            ['52 Week High', '52 Week Low', '52 Week Volatility', 'Last Year Return Rate']

            # ETF encoding
            etf_encoded = 1 if is_etf else 0

            # Aggregate features
            stock_features.append({
                'Ticker': ticker,
                'Company Name': company_name,
                'Market Cap': market_cap,
                'Sector': sector,
                'Industry': industry,
                'Open Price': open_price,
                'Close Price': close_price,
                'High Price': high_price,
                'Low Price': low_price,
                'Last Close': last_close,
                '52 Week High': high_52w,
                '52 Week Low': low_52w,
                'Last Month Volatility': last_month_volatility,
                '52 Week Volatility': year_volatility,
                'ETF': etf_encoded,
                'Yield to Date': yield_to_date,
                'Yearly Dividend Rate': dividend_rate,
                'Last Year Return Rate': last_year_return
            })
        except Exception as e:
            warnings.warn(f"Failed to retrieve data for {ticker}: {str(e)}")

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(stock_features)

    # Encode 'Sector' and 'Industry' as integer categories
    sector_mapping = encode_categories(df, 'Sector')
    industry_mapping = encode_categories(df, 'Industry')

    # Impute missing Market Cap using KNN (K=3)
    imputer = KNNImputer(n_neighbors=3)
    numerical_cols = ['Open Price', 'Close Price', 'High Price', 'Low Price', 'Last Close', '52 Week High', '52 Week Low',
                       'Last Month Volatility', '52 Week Volatility', 'Yield to Date', 'Yearly Dividend Rate', 'Last Year Return Rate']

    # Ensure that numerical columns are used for KNN imputation
    df['Market Cap'] = imputer.fit_transform(df[['Market Cap'] + numerical_cols])[:, 0]

    # Obtain names of columns with missing values 
    cols_with_missing = df.columns[df.isnull().any()].tolist() 

    # Perform KNN imputation for each column with missing values
    for col in cols_with_missing:
        # Columns to use for imputation: the current column + numerical columns
        imputation_cols = [col] + [c for c in numerical_cols if c != col]
        
        # Fit and transform on just the selected columns, keeping the imputed result for `col`
        df[col] = imputer.fit_transform(df[imputation_cols])[:, 0]

    # Remove ETF column and Yiled to Date if specified 
    if remove_etfs:
        df = df.drop('ETF', axis=1) # remove etf colum
        df = df.drop('Yield to Date', axis=1) # remove yield to date column

    # Recome duplicate columns 
    df = df.loc[:, ~df.columns.duplicated()]

    t1 = time.time() 
    print("Completed fetching stock data in {:.2f} seconds.".format(t1 - t0))

    # Save the data to a CSV file if a savepath is provided 
    if savepath:
        print("saving data to", savepath)
        df.to_csv(savepath, index=False)

    return df, sector_mapping, industry_mapping


def prepare_data_for_vae(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes the 'Sector' and 'Industry' columns in the input DataFrame.
    
    This function ensures that the one-hot encoded columns include all predefined 
    categories for 'Sector' and 'Industry' (including 'Other'), even if they are 
    not present in the data.
    
    Parameters:
    - df (DataFrame): The input DataFrame containing the 'Sector' and 'Industry' columns.
    
    Returns:
    - DataFrame: A new DataFrame with the 'Sector' and 'Industry' columns one-hot encoded.
    """
    # Define all possible categories for 'Sector' and 'Industry' including 'Other'
    all_sectors = SECTORS + ['Other']
    all_industries = INDUSTRIES + ['Other']

    # Ensure the columns exist in the DataFrame and contain only valid categories
    df['Sector'] = df['Sector'].where(df['Sector'].isin(all_sectors), 'Other')
    df['Industry'] = df['Industry'].where(df['Industry'].isin(all_industries), 'Other')

    # Drop irrelevant columns
    df = df.drop(['Ticker', 'Company Name'], axis=1)

    # Perform one-hot encoding with all predefined categories
    sector_dummies = pd.get_dummies(df['Sector'], prefix='Sector')
    industry_dummies = pd.get_dummies(df['Industry'], prefix='Industry')

    # Ensure all predefined categories are represented in the dummy columns
    for sector in all_sectors:
        if f"Sector_{sector}" not in sector_dummies.columns:
            sector_dummies[f"Sector_{sector}"] = 0

    for industry in all_industries:
        if f"Industry_{industry}" not in industry_dummies.columns:
            industry_dummies[f"Industry_{industry}"] = 0

    # Concatenate the one-hot encoded columns back to the DataFrame
    df = pd.concat([df, sector_dummies, industry_dummies], axis=1)

    # Drop the original 'Sector' and 'Industry' columns
    df = df.drop(['Sector', 'Industry'], axis=1)

    # Ensure all dummy columns are of type float
    for col in sector_dummies.columns.union(industry_dummies.columns):
        df[col] = df[col].astype(float)

    # Esure all columns are of type float64 
    df = df.astype(float)

    # Drop any remaining NA values
    df = df.dropna()

    return df


# # Example usage
# sp500_df = scrape_sp500_wikipedia()  # Use the function you created to scrape S&P 500 companies
# custom_tickers = ['TSLA', 'ZM', 'SNOW']  # Example custom tickers
# stock_data, sector_mapping, industry_mapping = fetch_stock_data(sp500_df, custom_tickers) # Fetch data
# stock_data_vae = prepare_data_for_vae(stock_data)  # Prepare data for VAE


