import pandas as pd
from config.config import ROOT_DIR


def parseInput():
    path = ROOT_DIR + '/files/training-Obama-Romney-tweets.xls'
    print(path)
    df = pd.read_excel(path)
    print(df)
