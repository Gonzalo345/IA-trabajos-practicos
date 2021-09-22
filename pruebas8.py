"""
K-means es uno de los algoritmos más básicos en Machine Learning no supervisado.
Es un algoritmo de clusterización, que agrupa datos que comparten características similares.
Recordemos que entendemos datos como n realizaciones del vector aleatorio X.

El algoritmo funciona de la siguiente manera:

 1_ El usuario selecciona la cantidad de clusters a crear n.
 2_ Se seleccionan n elementos aleatorios de X como posiciones iniciales del los centroides C.
 3_ Se calcula la distancia entre todos los puntos en X y todos los puntos en C.
 4_ Para cada punto en X se selecciona el centroide más cercano de C.
 5_ Se recalculan los centroides C a partir de usar las filas de X que pertenecen a cada centroide.
 6_ Se itera entre 3 y 5 una cantidad fija de veces o hasta que la posición de los centroides no cambie dada una
 tolerancia.

Se debe por lo tanto implementar la función k_means(X, n) de manera tal que, al finalizar, devuelva la posición de los
centroides y a qué cluster pertenece cada fila de X.

Hint: para (2) utilizar funciones de np.random, para (3) y (4) usar los ejercicios anteriores, para (5) es válido
utilizar un for. Iterar 10 veces entre (3) y (5).

datos = n realizaciones del vector aleatorio X

"""
import numpy as np
import matplotlib.pyplot as plt

print("\n---- Ejercicio 8 ----")
print("Implementación Básica de K-means\n")

n_cluster = 4
nv_overlap = 10
centroid = nv_overlap * np.random.random_sample((n_cluster, 2))
print(centroid)

n_samples = 200
data = np.repeat(centroid, n_samples / n_cluster, axis=0)
print(data)

normal_noise = np.random.normal(loc=0, scale=0.6, size=(len(data), 2))
data = data + normal_noise
print(data)

fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=120)
ax.scatter(data[:, 0], data[:, 1])
ax.set_xlim(-2, 12)
ax.set_ylim(-2, 12)
plt.show()

idx = np.random.choice(n_samples, n_cluster)
centroides_aleatorios = data[idx, :]
print(idx)
print(centroides_aleatorios)

colores = ["#00cc44",  # Verde
           "#ff7700",  # Naranja
           "#ff0000"  # Rojo
           ]

fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=120)
ax.scatter(data[:, 0], data[:, 1])
plt.scatter(centroides_aleatorios[:, 0], centroides_aleatorios[:, 1], c="#ff7700")
ax.set_xlim(-2, 12)
ax.set_ylim(-2, 12)
plt.show()


def dist(a, b):
    distancia = np.sum((a - b) ** 2, axis=1) ** (1 / 2)
    return distancia


def distancias_a_centroides(puntos, centroides):
    exppanded_C = centroides[:, None]
    distancias = np.sqrt(np.sum((exppanded_C - puntos) ** 2, axis=2))
    return distancias


dist_centroide = distancias_a_centroides(data, centroides_aleatorios)
arg_min = np.argmin(distancias_a_centroides(data, centroides_aleatorios), axis=0)
print(arg_min)
print(data[0])
