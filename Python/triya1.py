import numpy as np
import pandas as pd
import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

rs=150

df = pd.read_csv("E:\hepatitisHandling.csv")
clf = svm.SVC(gamma='scale')
bdt = AdaBoostClassifier(clf,algorithm="SAMME",n_estimators=20,random_state=rs)
sm = SMOTE(sampling_strategy='minority',random_state=rs)

'''df.drop(['number'],1,inplace=True)
print(df.shape)'''


X = np.array(df.drop(['Class'],1))
y = np.array(df['Class'])
X_res, y_res = sm.fit_resample(X, y)
print(X_res.shape)
n=len(y_res)
satu = []
dua = []
for i in range(n):
    if y_res[i] == 1:
        satu.append(y_res[i])
    else:
        dua.append(y_res[i])

print(len(satu))
print("===============")
print(len(dua))
"""
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.30, random_state=rs,shuffle=True)

start = time.time()
bdt.fit(X_train, y_train)
y_pred = bdt.predict(X_test)
end = time.time()

hasil = metrics.accuracy_score(y_test, y_pred)
print(hasil)
"""
print("=========== SVM ============")
start = time.time()
kf = KFold(n_splits=10, random_state=rs, shuffle=True)
hasil = cross_val_score(bdt, X, y, cv=kf, scoring='accuracy')
print(hasil.mean())
end = time.time()

print("=========== SVM+Smote ============")
kf = KFold(n_splits=10, random_state=rs, shuffle=True)
hasil2 = cross_val_score(bdt, X_res, y_res, cv=kf, scoring='accuracy') 
print(hasil2.mean())

waktu=end-start
print("======================")
