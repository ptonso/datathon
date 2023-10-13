import pandas as pd

def random_sample_imputation(df):
    '''
    impute missing values with random sample from non-missing value in same col
    arguments:
    dataframe with missing values
    returns:
    dataframe with no missing values
    '''
    
    cols_with_missing_values = df.columns[df.isna().any()].tolist()

    for var in cols_with_missing_values:

        # extract a random sample
        random_sample_df = df[var].dropna().sample(df[var].isnull().sum(), random_state=0)

        # re-index the randomply extracted sample
        random_sample_df.index = df[df[var].isnull()].index

        #replace the NA
        df.loc[df[var].isnull(), var] = random_sample_df

    return df