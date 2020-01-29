import pandas as pd
#import numpy as np
from sklearn.impute import SimpleImputer


df = pd.read_csv("D:\hepatitis\hepatitis_missing_num.csv")
imp = SimpleImputer(strategy='most_frequent')
imp.fit(df)
df = imp.transform(df)
df = pd.DataFrame(df)
df.to_csv("D:\hepatitis\hepatitis_most_num.csv")

