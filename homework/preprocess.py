#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pickle
import click
import pandas as pd

from sklearn.feature_extraction import DictVectorizer


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)
    
def load_raw_data(fileurl:str):
    df = pd.read_parquet(fileurl)
    return df
    
def remove_outlier(df: pd.DataFrame):
    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df['duration_mins']= (df['duration']).dt.total_seconds() / 60
    df= df[(df.duration_mins >= 1) & (df.duration_mins <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df
    
def preprocess_data(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv
    
@click.command()
@click.option(
    "--url",
    help="weblink where the raw NYC taxi trip data was saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
@click.option(
    "--year",
    help="Data year"
)
@click.option(
    "--train_month",
    help="month of the train data"
)
@click.option(
    "--val_month",
    help="month of the validation data"
)
@click.option(
    "--test_month",
    help="month of the test data"
)
def run_data_preprocess(url:str,dest_path:str,year: int,train_month: int, val_month: int,test_month: int,dataset: str = "green"):
    # Load parquet files
    df_train_dataset = load_raw_data(f"{url}/{dataset}_tripdata_{year}-{train_month}.parquet")
    df_validate_dataset = load_raw_data(f"{url}/{dataset}_tripdata_{year}-{val_month}.parquet")
    df_test_dataset = load_raw_data(f"{url}/{dataset}_tripdata_{year}-{test_month}.parquet")
    
    df_train = remove_outlier(df_train_dataset)
    df_validate = remove_outlier(df_validate_dataset)
    df_test = remove_outlier(df_test_dataset)
    
    # Extract the target
    target = 'duration_mins'
    y_train = df_train[target].values
    y_val = df_validate[target].values
    y_test = df_test[target].values
                            
    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess_data(df_train, dv, fit_dv=True)
    X_val, _ = preprocess_data(df_validate, dv, fit_dv=False)
    X_test, _ = preprocess_data(df_test, dv, fit_dv=False)
    
    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == '__main__':
    run_data_preprocess()


# In[ ]:




