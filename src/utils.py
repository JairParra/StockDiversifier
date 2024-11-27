"""
utils.py
    General utils functions for the project.

@author: Hair Parra
"""

################## 
### 1. Imports ###
##################

# Data Manipulation & Visualiztion  
import numpy as np

##########################
### 2. Utils Functions ###
##########################

# Convert ndarray values to lists for JSON serialization
def serialize_ndarray(data_dict):
    return {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in data_dict.items()}

# Convert lists back to ndarrays if needed
def deserialize_ndarray(data_dict):
    return {key: (np.array(value) if isinstance(value, list) else value) for key, value in data_dict.items()}