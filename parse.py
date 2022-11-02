import pandas as pd
from tabulate import tabulate

def parseInput():
    path = 'training-Obama-Romney-tweets.xlsx'
    df = pd.read_excel(path, sheet_name="Obama")
    #print(tabulate(df, headers='keys'))
    return df
