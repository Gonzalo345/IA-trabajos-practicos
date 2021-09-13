# -*- coding: utf-8 -*-
import numpy as np

"""
Sorting
Dada una matriz en formato numpy array, donde cada fila de la matriz representa un vector matemático, se requiere 
computar la norma l2 de cada vector. Una vez obtenida la norma, se debe ordenar las mismas de mayor a menor. Finalmente, 
obtener la matriz original ordenada por fila según la norma l2.
"""
print("\n---- Ejercicio 2 ----")
print("Computar la norma l2 de cada vector, ordenar de mayor a menor\n")

matriz_a = np.array([[5, 9, 7], [1, 2, -2], [3, 5, -6]])

print("Matriz A = \n", matriz_a)


def norma2(matriz):
    matriz = np.sum((matriz ** 2), axis=1) ** (1 / 2)
    return matriz


print("Norma 2 de la matriz A\n", norma2(matriz_a))
norma2_matriz_a_ordenada = -np.sort(-norma2(matriz_a))
print("Norma 2 ordenada de mayor a menor\n", norma2_matriz_a_ordenada)
