# %% Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling

# %% Lab Instructions
# Step one: Review these two datasets and brainstorm problems that 
# could be addressed with the dataset. Identify a question for each dataset.
# College Completion and Job Placement

# Step two: Work through the steps outlined in the examples to include the following elements:
# - Write a generic question that this dataset could address.
# - What is a independent Business Metric for your problem? Think about the case study examples we have discussed in class.
# - Data preparation:
#       - correct variable type/class as needed
#       - collapse factor levels as needed
#       - one-hot encoding factor variables
#       - normalize the continuous variables
#       - drop unneeded variables
#       - create target variable if needed
#       - Calculate the prevalence of the target variable
#       - Create the necessary data partitions (Train,Tune,Test)

# Step three: What do your instincts tell you about the data. 
# Can it address your problem, what areas/items are you worried about?

# Step four: Create functions for your two pipelines that produces the train 
# and test datasets. The end result should be a series of functions that can 
# be called to produce the train and test datasets for each of your two 
# problems that includes all the data prep steps you took. This is essentially 
# creating a DAG for your data prep steps. Imagine you will need to do this for 
# multiple problems in the future so creating functions that can be reused is 
# important. You don't need to create one full pipeline function that does 
# everything but rather a series of smaller functions that can be called in 
# sequence to produce the final datasets. Use your judgement on how to break 
# up the functions.
# %%
# College Completion Dataset
college = pd.read_csv("cc_institution_details.csv")
college_flagship = college[college['flagship'] == "X"]
college_6year_present = college[college['vsa_grad_after6_first'] > 0]
columns_percentiles = [col for col in college.columns  if "percentile" in col]

#%%
print(columns_percentiles)
# %%
# Central question: Can you predict 6-year, stayed the whole time graduation 
# rate based on other features?
# Column analysis
# Target column: vsa_grad_after6_first
# Identifier columns: Index, UnitID, Chron Name, Nicknames, Website, Similar. Drop these during training
# Large % missing columns that can be fixed: Historically black colleges (remap
# to boolean), flagship universities (remap to boolean)
# Large % missing columns without an obvious pattern: 
#   - All VSA columns, including graduation rates. Notably, after filtering
#     the colleges by the ones with 6-year grad rates available, they are
#     all 4-year public institutions.
# Categorical columns: city, state, level, control, basic, 
# Boolean columns: HBCU, Flagship
# Numeric columns: Latitude, longitude, student count, awards per graduate
#   columns, award expendiature columns, Full-time student columns, SAT columns,
#   aid columns, endowment columns, graduate percentile columns, pell columns,
#   student retainment columns, full-time faculty columns, VSA columns, state
#   sector count, Carnegie count, counted students percent, cohort size
# ** The percentile columns might not be necessary since everything will already
#    be scaled from 0-1
# Oddly formatted columns: counted_pct (Has two numbers, the second of which I'm
# unsure how to read), similar (seems like it should be a list but is one string),
#

#%% Cleanup Functions
def filter_columns_by_condition(df: pd.DataFrame, condition: bool = True, inv: bool = False):
    """Filters a data frame's columns by some condition.
    
    Args:
        df (DataFrame) - The frame to filter columns out of
        condition (boolean) - Some condition of format f(colname, colvalues) that returns a boolean. True by default
        inv (boolean) - Filters by the inverse of the condition. False by default
    Returns:
        DataFrame - The frame with columns filtered by the condition
    """
    cols_filtered = []
    for cname, cvals in df.items():
        if ((not condition(cname, cvals)) if inv else (condition(cname, cvals))):
            cols_filtered.append(cname)
    return df[cols_filtered]

def convert_col_to_boolean(df: pd.DataFrame, col: str, truthCondition: bool = False, nameOverride: str = ""):
    """Converts a data frame's column to Boolean, given some truth condition.
    
    Args:
        df (DataFrame) - The frame to convert a column into a boolean column
        col (str) - The name of the column to convert
        condition (boolean) - Some condition of format f(value) that returns a boolean. False by default
        nameOverride (str) - If provided, will replace the name of the old column. Empty by default
    """
    newCol = pd.Series(col, dtype=bool)
    for i in range(len(df)):
        newCol[i] = True if truthCondition(df[col][i]) else False
    df[col] = newCol
    if nameOverride:
        df.rename(columns={col: nameOverride}, inplace=True)
    

# %%
convert_col_to_boolean(college, "hbcu", lambda x: x == "X")
# %%
convert_col_to_boolean(college, "flagship", lambda x: x == "X")
# %%
convert_col_to_boolean(college, "control", lambda x: x == "Public", nameOverride="is_public")

# %%
