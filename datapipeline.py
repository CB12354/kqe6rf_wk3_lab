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
    """Downloads the college dataset and returns a formatted version of it.
    Returns:
        pd.DataFrame: a formatted version of the cc_institution_details dataset
    """
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
    """Applies ml_df_transformer() to the college information dataset.

    Args:
        college (pd.DataFrame): the DataFrame produced by format_college_dataset()
    Returns:
        pd.DataFrame: the transformed version of the college dataset
    """
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

def format_job_dataset():
    """Downloads the job placement dataset and returns a formatted version of it.
    Returns:
        pd.DataFrame: a formatted version of the job placement dataset
    """
    job = pd.read_csv(job_url)
    # Columns to convert to boolean: status -> is_employed, workex -> has_exp,
    # ssc_b / hsc_b -> ssc_in_central_board / hsc_in_central_board, gender -> is_male
    convert_col_to_boolean(job, 'status', lambda x: x == "Placed", "is_employed")
    convert_col_to_boolean(job, 'workex', lambda x: x == "Yes", "has_exp")
    convert_col_to_boolean(job, 'ssc_b', lambda x: x == "Central", "ssc_in_central_board")
    convert_col_to_boolean(job, 'hsc_b', lambda x: x == "Central", "hsc_in_central_board")
    convert_col_to_boolean(job, 'gender', lambda x: x == "M", "is_male")
    # Columns to fix: Salary (missing data should be 0)
    job['salary'] = job['salary'].apply(lambda x: x if x > 0 else 0)
    
    # Category columns that are not boolean: higher secondary specialization, 
    # Undergrad degree, MBA specialization
    cat_cols = ['hsc_s', 'degree_t', 'specialisation']
    for c in cat_cols:
        job[c] = job[c].astype('category')

def transform_job_dataset(job: DataFrame):
    """Applies ml_df_transformer() to the job placement dataset.

    Args:
        job (pd.DataFrame): the DataFrame produced by format_job_dataset()
    Returns:
        pd.DataFrame: the transformed version of the job dataset
    """
    # ID columns: Serial Number (drop during training)
    # Columns to drop: Salary (Salary post-placement is irrelevant in predicting placement)
    drop_cols = ['sl_no', 'salary']
    # Relatively normally distributed columns: ssc_p, hsc_p, degree_p, mba_p
    stand_cols = ['ssc_p', 'hsc_p', 'degree_p', 'mba_p']
    job_transformed = ml_df_transformer(job, drop_columns=drop_cols, standardize_columns=stand_cols)
    return job_transformed

# Helper function to get all 3 datasets at once
def triple_t_split(df: pd.DataFrame, tr_size, target):
    """Splits a dataset into 3: train, tune, and test

    Args:
        df (pd.DataFrame): The DataFrame to split
        train_size (float, int): Size of the training set. % if < 1, # if >=1 
        stratify_col (str): The column to stratify by. Preserves ratio of this column in
            the datasets

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: Train, tune, and test datasets
    """
    train, test = train_test_split(df, 
                                train_size=tr_size,
                                stratify=df[target])
    print(sum(train['is_generous']) / len(train))

    tune, test = train_test_split(test, 
                                train_size=0.5,
                                stratify=test[target])
    return train, tune, test

def college_t_sets(college: pd.DataFrame, tr_size):
    """Generates the training, tuning, and test sets for the college dataset.

    Args:
        college (pd.DataFrame): the result from transform_college_dataset()
        tr_size (float, int): The training set size. % if < 1, # if >=1 

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: training, tuning, and testing datasets
    """
    return triple_t_split(college, tr_size, 'is_generous')

def job_t_sets(job: pd.DataFrame, tr_size):
    """Generates the training, tuning, and test sets for the job placement dataset.

    Args:
        job (pd.DataFrame): the result from transform_job_dataset()
        tr_size (float, int): The training set size. % if < 1, # if >=1 

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: training, tuning, and testing datasets
    """
    return triple_t_split(job, tr_size, 'is_employed')