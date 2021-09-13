import numpy as np

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
