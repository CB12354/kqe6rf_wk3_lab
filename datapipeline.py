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

# %% Imports and setup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # For splitting data
from dataprep import *


# %% 
# Pipeline functions
def format_college_dataset():
    college = pd.read_csv("cc_institution_details.csv")
    # Collapse basic into top 5 categories + other
    top5basic = topNValues(college, "basic", 5)
    reduce_cat_col(college, "basic", top5basic)

    # Booleanize hbcu and flagship
    for col in ['hbcu', 'flagship']:
        convert_col_to_boolean(college, col, lambda x: x == "X")

    # Convert awards_per_value to is_generous
    convert_col_to_boolean(college, "awards_per_value", lambda x: x > np.percentile(college['awards_per_value'],75), "is_generous")


    # Filter out rows with less than 33.3% of students included in metrics (insufficient data)
    college['counted_pct'] = (college['counted_pct'].apply(lambda x: str(x).split("|")[0])).astype("float")
    college = college[college['counted_pct'] > 33.3].reset_index(drop=True)

    # Make US States into the main 4 us regions
    college['state'] = (college['state'].apply(region)).astype("category")
    college.rename(columns={'state' : 'region'},inplace=True)
    return college

def transform_college_dataset(college: DataFrame):
    # Transformation
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
    return college_transformed

def triple_t_split(df: pd.DataFrame, tr_size, target):
    """Splits a dataset into 3: train, tune, and test

    Args:
        df (DataFrame): The DataFrame to split
        train_size (float, int): Size of the training set. % if < 1, # if >=1 
        stratify_col (str): The column to stratify by. Preserves ratio of this column in
            the datasets

    Returns:
        DataFrame, DataFrame, DataFrame: Train, tune, and test datasets
    """
    train, test = train_test_split(df, 
                                train_size=tr_size,
                                stratify=df[target])
    print(sum(train['is_generous']) / len(train))

    tune, test = train_test_split(test, 
                                train_size=0.5,
                                stratify=test[target])
    return train, tune, test