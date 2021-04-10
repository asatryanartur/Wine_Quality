import pandas as pd

df = pd.read_csv("../winequalityN.csv")

df.to_csv('df.csv',index = False)
