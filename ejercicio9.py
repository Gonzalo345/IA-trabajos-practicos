"""
Ejercicio #9: Computar Métricas con __call__ house
En problemas de machine learning, es muy común que para cada predicción que obtenemos en nuestro dataset de verificación
y evaluación, almacenemos en arreglos de numpy el resultado de dicha predicción, junto con el valor verdadero y
parámetros auxiliares (como el ranking de la predicción y el query id).
Luego de obtener todas las predicciones, podemos utilizar la información almacenada en los arreglos de numpy, para
calcular todas las métricas que queremos medir en nuestro sistema.
Una buena práctica para implementar esto en Python, es crear clases que hereden de una clase Metric “base” y que cada
métrica implemente el método __call__.
Utilizar herencia, operador __call__ y kwargs, para escribir un programa que permita calcular todas las métricas de los
ejercicios anteriores mediante un for.
"""
import numpy as np
import matplotlib.pyplot as plt

print("\n---- Ejercicio 9 ----")
print("Computar Métricas con __call__ house\n")

