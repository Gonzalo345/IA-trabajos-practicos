# -*- coding: utf-8 -*-
import numpy as np

print("---- Ejercicio 1 ----")
print("Computar las normas l0, l1, l2, l-infinito\n")
matriz_a = np.array([[5, 9, 7],
                     [1, 2, -2],
                     [3, 5, -6]])

print("Matriz A = \n", matriz_a)


def norma0(matriz):
    """ Número de elementos diferentes a cero en el vector """
    matriz = (np.sum(matriz, axis=1))
    return matriz


def norma1(matriz):
    """ Norma 1, sumatoria de todos los componentes de un vector en su valor absoluto"""
    matriz = np.absolute(matriz)
    matriz = np.sum(matriz, axis=1)
    return matriz


def norma2(matriz):
    """ Norma 2, euclídea """
    matriz = np.sum((matriz ** 2), axis=1) ** (1 / 2)
    return matriz


def norma_infinito(matriz):
    print("Norma infinito, el maximo valor en el vector")
    matriz = np.amax(np.absolute(matriz))
    return matriz


print("Norma 0 sumatoria de elementos diferentes de cero en el vector       ", norma0(matriz_a))
print("Norma 1 sumatoria de valor absoluto de cada elemento en el vector    ", norma1(matriz_a))
print("Norma 2 raíz cuadrada de sumatoria de valores absolutos en el vector ", norma2(matriz_a))
print("Norma infinito, valor absoluto máximo en el vector                   ", norma_infinito(matriz_a))

