"Utility functions for preprocessing data."
import re
import pandas as pd
from IPython import embed


def preprocess_car_data(df):
    """Apply simple preprocessing and return cleaned data"""
    df.dropna(inplace=True)

    df["name"] = df["name"].str.split(" ").str[0]
    df["engine"] = df["engine"].str.split(" ").str[0]
    df["max_power"] = df["max_power"].str.split(" ").str[0]
    df["mileage"] = df["mileage"].str.split(" ").str[0]

    di = {"First Owner": 0.0 , "Second Owner": 1.0,"Third Owner": 2.0,
          "Fourth & Above Owner": 3.0, "Test Drive Car": 4.0}
    df.replace({"owner": di}, inplace=True)

    df.drop(["torque", "transmission", "seller_type", "fuel"], axis=1, inplace=True)

    return df.iloc[:, 1:].astype(float)


def car_train_test(df, train_split=0.8):
    """Create train and test datasets"""
    df = df.sample(frac=1).reset_index(drop=True)

    split_ind = int(train_split * len(df))
    df_train = df.iloc[:split_ind]
    df_test = df.iloc[split_ind:]

    return df_train, df_test

