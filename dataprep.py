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
college_sat = college[college['med_sat_value'] > 0]

# %%
# Central question: Say you're a private high school that wants to increase the 
# percent of their applicants that receive a merit scholarship in order to attract
# more intelligent students. Can you predict if a school is likely to give a
# student a merit scholarship using other metrics about the school?
# Independent business metric: Have students apply to schools with the most
# prevalent features in high-giving schools and see if the percent of recipients
# goes up compared to previous years
#  
# Column analysis
# Target column: awards_per_value. Will be made into a boolean column classifying
#   a "generous college" as one with a value above 22.
# Identifier columns: Index, UnitID, Chron Name, Nicknames, Website, Similar, 
#   Latitude, Longitude, Carnegie count. Drop these during training
# Large % missing columns that can be fixed: Historically black colleges (remap
# to boolean), flagship universities (remap to boolean)
# Unfixable large % missing columns: 
#   - All VSA columns, including graduation rates. Notably, after filtering
#     the colleges by the ones with 6-year grad rates available, they are
#     all 4-year public institutions. If we're predicting a broad range of colleges, they're unusable.
#   - SAT columns. It's mostly the well-documented colleges that have SAT data available which
#     biases against smaller/associates colleges.
# Categorical columns: city, state, level, control, basic
# Compressibles: State (Compress into main 4 US regions), Basic (Compress into top 4 or so categories + others)
# Removables: City (too many categories to be useful and cannot designate urban/rural easily)
# Boolean columns: HBCU, Flagship (currently Strings, need to convert)
# Numeric columns: Latitude, longitude, student count, awards per graduate
#   columns, award expendiature columns, Full-time student columns, SAT columns,
#   aid columns, endowment columns, graduate percentile columns, pell columns,
#   student retainment columns, full-time faculty columns, VSA columns, state
#   sector count, Carnegie count, counted students percent, cohort size
# ** The percentile columns might not be necessary since everything will already
#    be scaled from 0-1
# Oddly formatted columns: counted_pct. I'm not sure what the second number means.
#   I'm going to make it just the first number. 
#

#%% General Cleanup Functions
def filter_columns_by_condition(df: pd.DataFrame, condition: bool = True, inv: bool = False):
    """Filters a data frame's columns by some condition regarding its name or values.
    
    This function is meant to have a wider usability scope than Pandas' filter() function.
    
    Args:
        df (DataFrame): The frame to filter columns out of
        condition (boolean): Some condition of format f(colname, colvalues) that returns a boolean. True by default
        inv (boolean): Filters by the inverse of the condition. False by default
    Returns:
        DataFrame: The frame with columns filtered by the condition
    """
    cols_filtered = []
    for cname, cvals in df.items():
        if ((not condition(cname, cvals)) if inv else (condition(cname, cvals))):
            cols_filtered.append(cname)
    return df[cols_filtered]

def convert_col_to_boolean(df: pd.DataFrame, col: str, truthCondition: bool = False, nameOverride: str = ""):
    """Converts a data frame's column to Boolean, given some truth condition.
    
    Args:
        df (DataFrame): The frame to convert a column into a boolean column
        col (str): The name of the column to convert
        condition (boolean): Some condition of format f(value) that returns a boolean. False by default
        nameOverride (str): If provided, will replace the name of the old column. Empty by default
    """
    newCol = pd.Series(col, dtype=bool)
    for i in range(len(df)):
        newCol[i] = True if truthCondition(df[col][i]) else False
    df[col] = newCol
    if nameOverride:
        df.rename(columns={col: nameOverride}, inplace=True)

def reduce_cat_col(df: pd.DataFrame, col: str, acceptedCats: list):
    """Takes a column in a DataFrame, converts it to a Category type, and replaces values outside of acceptedCats with "Other".
    
    Args:
        df (DataFrame): The frame to operate on
        col (str): The name of the column to compress categories
        acceptedCats (list): Categories that will be preserved
    """
    df[col] = (df[col].apply(lambda x: x if x in acceptedCats 
                            else "Other")).astype("category")

def ml_df_transformer(df: pd.DataFrame, drop_columns: list = [], 
                      standardize_columns: list = []) -> pd.DataFrame:
    """Transforms a data frame to be usable by a k-means model.

    This function assumes that appropriate transformations of variables (like categories to booleans)
    have already been made. It iterates through the columns of a data frame and does the following:
    - If the column is in drop_columns, it is dropped
    - If the column is a string, it is converted to a category and one-hot encoded
    - If the column is already a category, it is one-hot encoded
    - If the column is numeric, it is either standardized if in standardize_columns
    or min-max scaled if not
    - If the column is boolean, it does nothing
    Args:
        df (pd.DataFrame): the data frame to transform
        drop_columns (list): columns to drop from the data frame
        standardize_columns (list): columns to use z-score instead of min-max scaling
    Returns:
        DataFrame: The transformed data frame
    """
    df_transformed = df.copy(deep = True)
    df_transformed = df_transformed.drop(labels=drop_columns, axis=1)
    for col in df_transformed.columns:
        print(col, df_transformed[col].dtype)
    return df_transformed

topNValues = lambda df, col, n: list(df[col].value_counts().index)[:n]
# %% Dataset-specific helpers
# US Region: designates an input state to its census region

def region(state):
    west = ["washington", "oregon", "california", "idaho", "montana","wyoming","utah","colorado",
        "nevada", "arizona","new mexico", "hawaii", "alaska"]
    midwest = ["north dakota", "south dakota", "nebraska", "kansas","minnesota", "iowa", "missouri",
            "wisconsin","illinois","michigan","indiana","ohio"]
    northeast = ["maine","new hampshire","vermont","massachusetts", "rhode island", "connecticut",
                "new york", "new jersey", "pennsylvania"]
    state = state.lower()
    if state in west:
        return "West"
    elif state in midwest:
        return "Midwest"
    elif state in northeast:
        return "Northeast"
    else:
        return "South" 


# %% Formatting
college = pd.read_csv("cc_institution_details.csv")
# Collapse basic into top 5 categories + other
top5basic = topNValues(college, "basic", 5)
reduce_cat_col(college, "basic", top5basic)

# Booleanize hbcu and flagship
for col in ['hbcu', 'flagship']:
    convert_col_to_boolean(college, col, lambda x: x == "X")

# Convert awards_per_value to is_generous. Makes a 44-56 true-false split
convert_col_to_boolean(college, "awards_per_value", lambda x: x > 22.0, "is_generous")

# Filter out rows with less than 33.3% of students included in metrics (insufficient data)
college['counted_pct'] = (college['counted_pct'].apply(lambda x: str(x).split("|")[0])).astype("float")
college = college[college['counted_pct'] > 33.3]

# Make US States into the main 4 us regions
college['state'] = (college['state'].apply(region)).astype("category")
college.rename(columns={'state' : 'region'},inplace=True)

# %%
# Transformation
drop_cols = ['index','unitid', 'chronname', 'similar', 'site', 'nicknames', 
             'long_x', 'lat_y', 'city', 'carnegie_ct', 
             'med_sat_value', 'med_sat_percentile']
# Drop VSA columns
for col in [og_col for og_col in college.columns if "vsa" in og_col]:
    drop_cols.append(col)
# Drop percentile columns
for col in [og_col for og_col in college.columns if "percentile" in og_col]:
    drop_cols.append(col)
college_transformed = ml_df_transformer(college, 
                                        drop_columns=drop_cols, 
                                        standardize_columns=[])


# %%
