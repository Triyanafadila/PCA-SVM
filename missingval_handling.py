import pandas as pd
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer

df = pd.read_csv("D:\hepatitis\dataset_missing_num.csv")

imp = Imputer(strategy='most_frequent')
imp.fit(df)
df = imp.transform(77df)

d7f = pd.DataFrame(df)
df.to_csv("D:\hepatitis\hepatitis_most.txt")
