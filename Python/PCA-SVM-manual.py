import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib import pyplot as plt

df = pd.read_csv("D:\Triyana\hepatitis\hepatitis01.csv")
# Separating out the features
X = np.array(df.drop(['Class'],1))
# Separating out the target
y = np.array(df['Class'])

#normalisasi (to have zero-mean and unit-variance such that each feature will be weighted equally in our calculations).
X = StandardScaler().fit_transform(X)

# Compute the mean of the data
mean_vec = np.mean(X, axis=0)
# Compute the covariance matrix
#cov_mat = (X - mean_vec).T.dot((X - mean_vec)) / (X.shape[0]-1)
# OR we can do this with one line of numpy:
cov_mat = np.cov(X.T)

# Compute the eigen values and vectors using numpy
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvector \n%s'%eig_vecs)
print('Eigenvalues \n%s'%eig_vals)

# Make a list of (eigenvalue, eigenvector) tuples and sort from high to low
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
print('Eigenvalues descending order: ')
for i in eig_pairs:
	print(i[0])

# Only keep a certain number of eigen vectors based on the "explained variance percentage"
#which tells us how much information (variance) can be attributed to each of the principal components
exp_var_percentage = 0.80 # Threshold of 97% explained variance
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
num_vec_to_keep = 0
for index, percentage in enumerate(cum_var_exp):
	if percentage > exp_var_percentage:
		num_vec_to_keep = index + 1
	break
print('explained varianced',var_exp)
print('jumlah persen explained varianced',cum_var_exp)

# Compute the projection matrix based on the top eigen vectors
num_features = X.shape[1]
proj_mat = eig_pairs[0][1].reshape(num_features,1)
for eig_vec_idx in range(1, num_vec_to_keep):
 proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features,1)))

# Project the data
pca_data = X.dot(proj_mat)

X_train,X_test,y_train,y_test = train_test_split(pca_data,y,test_size=0.2, random_state=30)
clf = SVC(C=1.0,  gamma='auto', random_state=30)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test,y_pred))
skor = accuracy_score(y_test, y_pred)
print('akurasi:', skor)
