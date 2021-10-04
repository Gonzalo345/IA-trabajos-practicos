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

# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline

print("\n---- Ejercicio 8 ----")
print("Implementación Básica de K-means\n")


def initialize_clusters(points, k):
    """Initializes clusters as k randomly selected points from points."""
    return points[np.random.randint(points.shape[0], size=k)]


# Function for calculating the distance between centroids
def get_distances(centroid, points):
    """Returns the distance the centroid is from each data point in points."""
    return np.linalg.norm(points - centroid, axis=1)


def make_points(centers, n_samples, nv_overlap):
    n_cluster = centers
    centroid = nv_overlap * np.random.random_sample((n_cluster, 2))
    points = np.repeat(centroid, n_samples / n_cluster, axis=0)
    normal_noise = np.random.normal(loc=0, scale=0.9, size=(len(points), 2))
    points = points + normal_noise
    classes = np.zeros(n_samples, dtype=np.float64)
    return points, classes


# Generate dataset
X, y = make_points(centers=4, n_samples=200, nv_overlap=20)

# Visualize
fig, ax = plt.subplots(figsize=(4, 4))

ax.scatter(X[:, 0], X[:, 1])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
plt.show()

k = 4

maxiter = 50


# Initialize our centroids by picking random data points
centroids = initialize_clusters(X, k)

# Initialize the vectors in which we will store the
# assigned classes of each data point and the
# calculated distances from each centroid
classes = np.zeros(X.shape[0], dtype=np.float64)
distances = np.zeros([X.shape[0], k], dtype=np.float64)

# Loop for the maximum number of iterations
for i in range(maxiter):

    # Assign all points to the nearest centroid
    for i, c in enumerate(centroids):
        distances[:, i] = get_distances(c, X)

    # Determine class membership of each point
    # by picking the closest centroid
    classes = np.argmin(distances, axis=1)

    # Update centroid location using the newly
    # assigned data point classes
    for c in range(k):
        centroids[c] = np.mean(X[classes == c], 0)


group_colors = ['skyblue', 'coral', 'lightgreen', 'salmon']
colors = [group_colors[j] for j in classes]

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(X[:, 0], X[:, 1], color=colors)
ax.scatter(centroids[:, 0], centroids[:, 1], color=['blue', 'orange', 'green', 'red'], marker='o', lw=2)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
plt.show()

