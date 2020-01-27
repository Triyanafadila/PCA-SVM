import pandas as pd
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer

df = pd.read_csv("D:\hepatitis\dataset_missing_num.csv")
7
imp = Imputer(strategy='most_frequent')
imp.fit(df)
df = imp.transform(df)

df = pd.DataFrame(df)
df.to_csv("D:\hepatitis\hepatitis_most.txt")
7