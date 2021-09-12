# -*- coding: utf-8 -*-
import numpy as np

print("---- Ejercicio 1 ----")
print("Computar las normas l0, l1, l2, l-infinito\n")
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


print("Norma 0", norma0(matriz_a))
print("Norma 1", norma1(matriz_a))
print("Norma 2", norma2(matriz_a))
print("Norma inf", norma_infinito(matriz_a))


print("\n---- Ejercicio 4 ----")
print("Precision, Recall, Accuracy\n")
print("True Positive (TP): la verdad es 1 y la predicción es 1.")
print("True Negative (TN): la verdad es 0 y la predicción es 0.")
print("False Negative (FN): la verdad es 1 y la predicción es 0.")
print("False Positive (FP): la verdad es 0 y la predicción es 1.\n")
'''--
En los problemas de clasificación, se cuenta con dos arreglos, la verdad (ground truth) y la predicción (prediction). 
Cada elemento de los arreglos puede tomar dos valores: 
True (representado por 1) y False (representado por 0). 
Por lo tanto, se pueden definir cuatro variables:   
     
True Positive (TP): la verdad es 1 y la predicción es 1.
True Negative (TN): la verdad es 0 y la predicción es 0.
False Negative (FN): la verdad es 1 y la predicción es 0.
False Positive (FP): la verdad es 0 y la predicción es 1.
A partir de esas cuatro variables, se definen las siguientes métricas:

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Para los siguientes arreglos, representando la verdad y la predicción, calcular las métricas anteriores con 
operaciones vectorizadas en NumPy.

truth = [1,1,0,1,1,1,0,0,0,1]
prediction = [1,1,1,1,0,0,1,1,0,0] '''

print("Vectores : \n ")
truth = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1])  # Verdad
print("Verdadero ", truth)
prediction = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0])  # Predicción
print("Predicción", prediction, "\n")
# valor esperado para TP ([1 1 0 1 0 0 0 0 0 0])
# valor esperado para TN ([0 0 0 0 0 0 0 0 1 0])
# valor esperado para TP ([0 0 0 0 1 1 0 0 0 1])
# valor esperado para TN ([0 0 1 0 0 0 1 1 0 0])

print("Valor esperado para True Positive ", "[1 1 0 1 0 0 0 0 0 0]")
truePositive = (truth == 1) & (prediction == 1)
# truePositive =  truth & prediction
print("Valor obtenido para True Positive ", truePositive.astype(int))

print("Valor esperado para True Negative ", "[0 0 0 0 0 0 0 0 1 0]")
trueNegative = (truth == 0) & (prediction == 0)
# trueNegative =  truth & prediction
print("Valor obtenido para True Negative ", trueNegative.astype(int))

print("Valor esperado para False Negative", "[0 0 0 0 1 1 0 0 0 1]")
falseNegative = (truth == 1) & (prediction == 0)
print("Valor obtenido para False Negative", falseNegative.astype(int))

print("Valor esperado para False Positive", "[0 0 1 0 0 0 1 1 0 0]")
falsePositive = (truth == 0) & (prediction == 1)
print("Valor obtenido para False Positive", falsePositive.astype(int))

precision = np.sum(truePositive) / (np.sum(truePositive) + np.sum(falsePositive))
print("Precision = TP / (TP + FP) = ", precision)

recall = np.sum(truePositive) / (np.sum(truePositive) + np.sum(falseNegative))
print("Recall = TP / (TP + FN) = ", recall)

accuracy = (np.sum(truePositive) + np.sum(trueNegative)) / (
            np.sum(truePositive) + np.sum(trueNegative) + np.sum(falsePositive) + np.sum(falseNegative))
print("Accuracy = (TP + TN) / (TP + TN + FP + FN) = ", accuracy)

print("\n---- Ejercicio 5 ----")
print("Average Query Precisión\n")

'''En information retrieval o search engines, en general contamos con queries “q” y para cada “q” una lista de documentos 
que son verdaderamente relevantes. Para evaluar un search engine, es común utilizar la métrica average query precision. 
Tomando de referencia el siguiente ejemplo, calcular la métrica con NumPy utilizando operaciones vectorizadas.

q_id =             [1, 1, 1, 1,  2, 2, 2,  3, 3, 3, 3, 3,  4, 4, 4, 4]
predicted_rank =   [0, 1, 2, 3,  0, 1, 2,  0, 1, 2, 3, 4,  0, 1, 2, 3]
truth_relevance =  [T, F, T, F,  T, T, T,  F, F, F, F, F,  T, F, F, T] 

Precision para q_id 1 = 2 / 4
Precision para q_id 2 = 3 / 3
Precision para q_id 3 = 0 / 5
Precision para q_id 4 = 2 / 4

average query precision = ((2/4) + (3/3) + (0/5) + (2/4)) / 4
'''

q_id = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4])
predicted_rank = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3])
# predicted_rank  = np.array([1, 0, 1, 0,      1, 1, 1,       0, 0, 0, 0, 0,       1, 0, 0, 1])
truth_relevance = np.array(['T', 'F', 'T', 'F', 'T', 'T', 'T', 'F', 'F', 'F', 'F', 'F', 'T', 'F', 'F', 'T'])

print("q_id             = ", q_id)
print("Predicted_rank   = ", predicted_rank)
print("Truth_relevance  = ", truth_relevance)

truth_relevance = (truth_relevance == 'T')

'''Creo que tengo que contar la cantidad de T por cada id
'''
auxTrue = np.zeros(np.amax(predicted_rank), dtype=int)
auxDatos = np.zeros(np.amax(predicted_rank), dtype=int)
for i in range(q_id.size):
    if truth_relevance[i]:
        auxTrue[q_id[i] - 1] += 1
    auxDatos[q_id[i] - 1] += 1
print("auxTrue", auxTrue)
print("auxDatos", auxDatos)
for i in range(np.amax(predicted_rank)):
    print("Precision para q_id ", i + 1, " = ", auxTrue[i], " / ", auxDatos[i], " = ", auxTrue[i] / auxDatos[i])

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

print("\n---- Ejercicio 7 ----")
print("Etiquetar Cluster\n")

'''
Obtener para cada fila en X, el índice de la fila en C con distancia euclídea más pequeña. 
Es decir, para cada fila en X, determinar a qué cluster pertenece en C. 
Hint: usar np.argmin.
'''
print("x = \n", X)
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
