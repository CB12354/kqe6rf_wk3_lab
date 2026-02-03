# %% Imports and setup
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype as is_ndt
from pandas.api.types import is_string_dtype as is_sdt
from pandas.api.types import is_bool_dtype as is_bdt
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
np.random.seed(2032026)

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


# %% Load Datasets for EDA
# College Completion Dataset
college = pd.read_csv("cc_institution_details.csv")
# Job Placement Dataset
job_url = "https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv"
job = pd.read_csv(job_url)

# %%
# COLLEGE COMPLETION DATASET
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
#   - Endowment value. Too much missing seemingly at random (less % than most missing)
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
# There's a huge base of missing data that makes it hard to predict things on a wide scope.
#   If you wanted to make a question about 4-year public institutions specifically, you'd be
#   in good hands. They have the most data available. In terms of my question, which has a
#   broader scope, there's a lot of columns you need to filter out simply because there's not
#   enough information for the schools with scholarship data available (ex. the VSA columns). 
#   This dataset could still answer the scholarship question, but with an incomplete picture 
#   of the statistics.

# %%
# JOB PLACEMENT DATASET
# Central question: What feature is most important in predicting job placement?
# Independent business metric: Say you're a company like Indeed that tries to connect
# employees to employers. If a factor rises to the surface, try to more aggressively
# market employees with that factor to employers. Metric is hiring rates, if they go
# up after implementing that strategy then the model did its job.
#
# Column analysis: 
# Target column: Status
# ID columns: sl-no (drop during training)

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
    mms = MinMaxScaler()
    ss = StandardScaler()
    cat_cols = []
    for col in df_transformed.columns:
        if is_bdt(df_transformed[col]):
            pass
        elif is_ndt(df_transformed[col]):
            if col in standardize_columns:
                df_transformed[col] = ss.fit_transform(df_transformed[[col]])
            else:
                df_transformed[col] = mms.fit_transform(df_transformed[[col]])
        elif is_sdt(df_transformed[col]) or df_transformed[col].dtype == 'category':
            df_transformed[col] = df_transformed[col].astype('category')
            cat_cols.append(col)
    df_transformed = pd.get_dummies(df_transformed, columns=cat_cols)
            
    df_transformed = df_transformed.dropna()
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


# %% Formatting college data set
college = pd.read_csv("cc_institution_details.csv")
# Collapse basic into top 5 categories + other
top5basic = topNValues(college, "basic", 5)
reduce_cat_col(college, "basic", top5basic)

# Booleanize hbcu and flagship
for col in ['hbcu', 'flagship']:
    convert_col_to_boolean(college, col, lambda x: x == "X")

# Convert awards_per_value to is_generous
convert_col_to_boolean(college, 
                       "awards_per_value", 
                       lambda x: x > np.percentile(college['awards_per_value'],75), 
                       "is_generous")


# Filter out rows with less than 33.3% of students included in metrics (insufficient data)
college['counted_pct'] = (college['counted_pct'].apply(lambda x: str(x).split("|")[0])).astype("float")
college = college[college['counted_pct'] > 33.3].reset_index(drop=True)

# Make US States into the main 4 us regions
college['state'] = (college['state'].apply(region)).astype("category")
college.rename(columns={'state' : 'region'},inplace=True)

# %% Transform college dataset

drop_cols = ['index','unitid', 'chronname', 'similar', 'site', 'nicknames', 
             'long_x', 'lat_y', 'city', 'carnegie_ct', 
             'med_sat_value', "endow_value"]
# Drop VSA columns
for col in [og_col for og_col in college.columns if "vsa" in og_col]:
    drop_cols.append(col)
# Drop percentile columns
for col in [og_col for og_col in college.columns if "percentile" in og_col]:
    drop_cols.append(col)
college_transformed = ml_df_transformer(college, 
                                        drop_columns=drop_cols, 
                                        standardize_columns=['grad_150_value','pell_value',
                                                             'retain_value',''])

# Calculate prevalence of generous colleges in transformed dataset
print(sum(college_transformed['is_generous']) / len(college_transformed))
# about 16% 

# %% Train and test datasets for college dataset 
train, test = train_test_split(college_transformed, 
                               train_size=0.6,
                               stratify=college_transformed['is_generous'])
print(sum(train['is_generous']) / len(train))

tune, test = train_test_split(test, 
                               train_size=0.5,
                               stratify=test['is_generous'])
