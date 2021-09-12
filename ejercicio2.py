# -*- coding: utf-8 -*-
import numpy as np

print("\n---- Ejercicio 2 ----")
print("Computar la norma l2 de cada vector, ordenar de mayor a menor\n")

matriz_a = np.array([[5, 9, 7], [1, 2, -2], [3, 5, -6]])

print("Matriz A = \n", matriz_a)


def norma0(matriz):
    matriz = (np.sum(matriz, axis=1))
    return matriz


def norma1(matriz):
    matriz = np.absolute(matriz)
    matriz = np.sum(matriz, axis=1)
    return matriz


def norma2(matriz):
    matriz = np.sum((matriz ** 2), axis=1) ** (1 / 2)
    return matriz


def norma_infinito(matriz):
    matriz = np.amax(np.absolute(matriz))
    return matriz


print("Matriz B = \n", matriz_a)
print("Norma 2\n", norma2(matriz_a))
b = -np.sort(-norma2(matriz_a))
print("Norma 2 ordena\n", b)
