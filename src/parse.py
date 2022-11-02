import pandas as pd
from config.config import ROOT_DIR
from tabulate import tabulate

def parseInput():
    path = ROOT_DIR + '/files/training-Obama-Romney-tweets.xlsx'
    df = pd.read_excel(path, sheet_name="Obama")
    #print(tabulate(df, headers='keys'))
    return df
