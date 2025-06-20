import pandas as pd

def filter_df(df, filter_list):
    filtered_df = df[~df['url'].isin(filter_list)]
    return filtered_df