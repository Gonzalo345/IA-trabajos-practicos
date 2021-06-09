# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 20:02:08 2021

@author: gonza
"""

'''
K-means es uno de los algoritmos más básicos en Machine Learning no supervisado. 
Es un algoritmo de clusterización, que agrupa datos que comparten características similares. 
Recordemos que entendemos datos como n realizaciones del vector aleatorio X.

El algoritmo funciona de la siguiente manera:

 1_ El usuario selecciona la cantidad de clusters a crear n. 
 2_ Se seleccionan n elementos aleatorios de X como posiciones iniciales del los centroides C.
 3_ Se calcula la distancia entre todos los puntos en X y todos los puntos en C.
 4_ Para cada punto en X se selecciona el centroide más cercano de C.
 5_ Se recalculan los centroides C a partir de usar las filas de X que pertenecen a cada centroide.
 6_ Se itera entre 3 y 5 una cantidad fija de veces o hasta que la posición de los centroides no cambie dada una tolerancia.
 
Se debe por lo tanto implementar la función k_means(X, n) de manera tal que, al finalizar, devuelva la posición de los 
centroides y a qué cluster pertenece cada fila de X.

Hint: para (2) utilizar funciones de np.random, para (3) y (4) usar los ejercicios anteriores, para (5) es válido 
utilizar un for. Iterar 10 veces entre (3) y (5).
'''

'''
datos = n realizaciones del vector aleatorio X

'''

print("\n---- Ejercicio 8 ----")
print("Implementación Básica de K-means\n")

import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


MAX_ITERATIONS = 5

def k_means(X, n_clusters): # el usuario selecciona la cantidad de clusters a crear n 
    '''
    numpy.eye(N, M=None, k=0, dtype=<class 'float'>, order='C', *, like=None)
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    '''
    centroids = np.eye(n_clusters, X.shape[1])          # Se seleccionan n elementos aleatorios como posiciones iniciales de los centroides
    print("Centroides \n", centroids)
    for i in range(MAX_ITERATIONS):
        print("Iteration # {}".format(i))
        centroids, cluster_ids = k_means_loop(X, centroids)
        print(centroids)
    return centroids, cluster_ids

def k_means_loop(X, centroids):
    
    # encontrar el label de cada fila de X en funcion de los centroides
    expanded_centroids = centroids[:, None]
    distances = np.sqrt(np.sum((expanded_centroids - X) ** 2, axis=2))
    arg_min = np.argmin(distances, axis=0)
    #print("Distancias \n", distances)
    
    #rederterminar los centroides
    for i in range(centroids.shape[0]):
        centroids[i] = np.mean(X[arg_min == i, :], axis=0)
    #print("Centroides 2\n", centroids)    
    return centroids, arg_min



class SyntheticDataset(object):
    
    def __init__(self, n_samples, inv_overlap):
        self.n_samples = n_samples
        self.inv_overlap = inv_overlap
        self.data, self.cluster_ids = self._build_cluster()
        
    def train_valid_slpit(self):
        idxs = np.random.permutation(self.n_samples)
        n_train_samples = int(self.n_samples * 0.8)
        train = self.data[idxs[:n_train_samples]]
        train_cluster_ids = self.cluster_ids[idxs[:n_train_samples]]
        valid = self.data[idxs[n_train_samples:]]
        valid_cluster_ids = self.cluster_ids[idxs[n_train_samples:]]
        return train, train_cluster_ids, valid, valid_cluster_ids
    
    @staticmethod
    def reduce_dimension(data, n_components):
        data_std = StandardScaler().fit_transform(data)
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data_std)
        
    def _build_cluster(self):
        centroids = np.array([
            [1, 0, 0, 0,],
            [0, 1, 0, 0,],
        ], dtype=np.float32)
        centroids = centroids * self.inv_overlap
        data = np.repeat(centroids, self.n_samples / 2, axis=0)
        #s = np.random.normal(mu, sigma, 1000)
        normal_noise = np.random.normal(loc=0, scale=1, size=(self.n_samples, 4))
        data = data + normal_noise
        
        cluster_ids = np.array([
            [0],
            [1],
        ])
        cluster_ids = np.repeat(cluster_ids, self.n_samples / 2, axis=0)
        return data, cluster_ids


if __name__ == '__main__':
    synthetic_dataset = SyntheticDataset(n_samples = 10, inv_overlap = 18)
    #with open('', 'wb') as file:
        #pickle.dump(synthetic_dataset, file)
    print("Creamos el dataset")
    data, cluster_ids = synthetic_dataset._build_cluster()
    print("data = \n", data)
    centroids, cluster_ids = k_means(data, n_clusters=2)

print("Nuevos centroides \n", centroids)
print("Pertenencia de los puntos  a los respectivos centroides\n", cluster_ids)        
        
        
        
        
        