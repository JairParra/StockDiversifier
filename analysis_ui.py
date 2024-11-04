"""
analysis_ui.py
    This script contains functions for visualizing and analyzing the recommendations from the Beta-VAE recommender model.

@author: Hair Parra
@author: Olivier Makuch
"""

################## 
### 1. Imports ###
##################

# Import and configure streamlit 
import streamlit as st

# Set the overall layout to be wider
st.set_page_config(layout="wide")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# set sns style 
sns.set_theme()

# setup dark plotting
sns.set_theme(style="darkgrid", context="talk")
plt.style.use("dark_background")
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})

################## 
### 2. Helpers ###
##################

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('data_raw/wsb_clean.csv')


