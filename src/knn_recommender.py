import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors



def prepare_anime_data(df):
    df_final = df.copy()

    # drop anime w/ nan for genre and type
    df_final = df_final.dropna(subset=['genre','type'])

    # drop ratings? episode count?
    # feature engineering: dummify genre and type



    return df_final


