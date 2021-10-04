import numpy as np

print("\n---- Ejercicio 7 ----")
print("Etiquetar Cluster\n")

'''
Obtener para cada fila en X, el índice de la fila en C con distancia euclídea más pequeña. 
Es decir, para cada fila en X, determinar a qué cluster pertenece en C. 
Hint: usar np.argmin.

X = [[1, 2, 3], 
     [4, 5, 6], 
     [7, 8, 9]]
C = [[1, 0, 0], 
     [0, 1, 1]]   
'''
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Puntos     = \n", X)
C = np.array([[1, 0, 0], [0, 1, 1]])
print("Centroides = \n", C)

'''Obtener para cada punto en X el cluster mas cercano'''

print("Valores calculados ej 6 =  [ 3.6  8.3  13.4]")
print("                           [ 2.4  7.5  12.7]")
print("Resultado esperado         [ P[1] P[1] P[1]]")


def distancias_a_centroides(puntos, centroides):
    '''
    Parameters
     numpy.reshape(a, newshape, order='C')
    '''
    exppanded_C = centroides[:, None]
    distancias = np.sqrt(np.sum((exppanded_C - puntos) ** 2, axis=2))
    print("Distancias \n", distancias)
    return distancias


arg_min = np.argmin(distancias_a_centroides(X, C), axis=0)

print("Centroide de pertenencia \n", arg_min)