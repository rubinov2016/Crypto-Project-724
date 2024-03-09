import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_distribution(stock_symbol, df):
    if df.index.name is None or df.index.name != df.columns[0]:
        df = df.set_index(df.columns[0])
    df = df.transpose()