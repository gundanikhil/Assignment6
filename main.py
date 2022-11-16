import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

dataset = pd.read_csv('E:/CC GENERAL.csv')
print(dataset.head())
print(dataset.shape)
print(dataset.isnull().sum())
print(dataset.shape)

dataset.drop(['CREDIT_LIMIT'], axis=1,inplace=True)
print(dataset.isnull().sum())

dataset['MINIMUM_PAYMENTS'] = dataset['MINIMUM_PAYMENTS'].replace(np.NaN, dataset['MINIMUM_PAYMENTS'].mean())
print(dataset.isnull().sum())

x = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1]
print(x.shape,y.shape)

scaler = StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)

normal = normalize(x)
print('Normalised data =', normal)

pca = PCA(n_components=2)

normal1 = pca.fit_transform(normal)
normal1 = pd.DataFrame(normal1)
normal1.columns = ['x1', 'x2']


plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram = shc.dendrogram((shc.linkage(normal1, method ='ward')))

ac2 = AgglomerativeClustering(n_clusters = 2)

plt.figure(figsize =(6, 6))
plt.scatter(normal1['x1'], normal1['x2'],c = ac2.fit_predict(normal1), cmap ='rainbow')
plt.title('K = 2')
plt.show()

ac3 = AgglomerativeClustering(n_clusters = 3)

plt.figure(figsize =(6, 6))
plt.scatter(normal1['x1'], normal1['x2'],c = ac3.fit_predict(normal1), cmap ='rainbow')
plt.title('K = 3')
plt.show()


ac4 = AgglomerativeClustering(n_clusters = 4)

plt.figure(figsize =(6, 6))
plt.scatter(normal1['x1'], normal1['x2'],c = ac4.fit_predict(normal1), cmap ='rainbow')
plt.title('K = 4')
plt.show()

ac5 = AgglomerativeClustering(n_clusters = 5)


plt.figure(figsize =(6, 6))
plt.scatter(normal1['x1'], normal1['x2'],c = ac5.fit_predict(normal1), cmap ='rainbow')
plt.title('K = 5')
plt.show()

k = [2, 3, 4, 5]

silhouette_scores = []
silhouette_scores.append(silhouette_score(normal1, ac2.fit_predict(normal1)))
silhouette_scores.append(silhouette_score(normal1, ac3.fit_predict(normal1)))
silhouette_scores.append(silhouette_score(normal1, ac4.fit_predict(normal1)))
silhouette_scores.append(silhouette_score(normal1, ac5.fit_predict(normal1)))



plt.bar(k, silhouette_scores)
plt.xlabel('Number of clusters', fontsize = 20)
plt.ylabel('S(i)', fontsize = 20)
plt.show()





