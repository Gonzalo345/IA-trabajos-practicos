import numpy as np

print("\n---- Ejercicio 6 ----")
print("Distancia a Centroides\n")

'''Dada una nube de puntos X y centroides C, obtener la distancia entre cada vector X y los centroides utilizando 
operaciones vectorizadas y broadcasting en NumPy. Utilizar como referencia los siguientes valores: 

X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
C = [[1, 0, 0], [0, 1, 1]]   
'''


def dist(a, b):
    distancia = np.sum((a - b) ** 2, axis=1) ** (1 / 2)
    return distancia


X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Puntos     = \n", X)
C = np.array([[1, 0, 0], [0, 1, 1]])
print("Centroides = \n", C)

print("Distancia de los puntos en X a C[0] es = ", dist(X, C[0]))
print("Distancia de los puntos en X a C[1] es = ", dist(X, C[1]))