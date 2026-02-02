# %% Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling

# %%
# College Completion Dataset
college = pd.read_csv("cc_institution_details.csv")
# %%
