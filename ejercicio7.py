# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:19:43 2021

@author: gonza
"""
import numpy as np

MAX_ITERATIONS = 10

print("\n---- Ejercicio 8 ----")
print("Implementación Básica de K-means\n")

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

def k_means(X, n_clusters):
    '''
    numpy.eye(N, M=None, k=0, dtype=<class 'float'>, order='C', *, like=None)
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    '''
    centroids = np.eye(n_clusters, X.shape[1])
    print(centroids)
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
    # redeterminar los centroides
    for i in range(centroids.shape[0]):
        centroids[i] = np.mean(X[arg_min == i, :], axis=0)
        
k_means()
        